"""
api.py — FastAPI backend for the cds_textagent CDSS
=====================================================
Exposes three endpoints:
  GET  /health        — liveness probe
  POST /chat          — runs the full LangGraph multi-agent pipeline
  POST /lookup-case   — direct ChromaDB metadata lookup by case_id / article_id
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level graph handles (populated at startup)
# ---------------------------------------------------------------------------

_langgraph_app = None  # type: ignore[var-annotated]
_create_initial_state = None  # type: ignore[var-annotated]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _langgraph_app, _create_initial_state
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("[API] Loading cds_textagent graph…")
    from cds_textagent.graph import app as _app
    from cds_textagent.state import create_initial_state
    _langgraph_app = _app
    _create_initial_state = create_initial_state
    logger.info("[API] Graph ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CDSS Text Agent API",
    description="Clinical Decision Support System backed by a LangGraph multi-agent pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    agents_used: list[str]
    elapsed_seconds: float


class CaseLookupRequest(BaseModel):
    case_id: str
    article_id: str = ""


class CaseDocument(BaseModel):
    case_id: str
    article_id: str
    content: str
    metadata: dict


class CaseLookupResponse(BaseModel):
    query_id: str
    results: list[CaseDocument]
    total_found: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "cds_textagent"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run the full multi-agent CDSS pipeline and return the synthesised answer."""
    if _langgraph_app is None or _create_initial_state is None:
        raise HTTPException(status_code=503, detail="Graph not yet initialised.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    session_id = request.session_id or str(uuid.uuid4())
    initial_state = _create_initial_state(query=question, session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    t0 = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _langgraph_app.invoke(initial_state, config=config),  # type: ignore[union-attr]
        )
    except Exception as exc:
        logger.exception("[API] Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.perf_counter() - t0, 2)

    return ChatResponse(
        answer=result.get("final_answer", ""),
        session_id=session_id,
        agents_used=list(dict.fromkeys(result.get("agents_used", []))),
        elapsed_seconds=elapsed,
    )


@app.post("/lookup-case", response_model=CaseLookupResponse)
async def lookup_case(request: CaseLookupRequest):
    """Direct ChromaDB metadata lookup by case_id (and optionally article_id).
    Does NOT run the LangGraph pipeline — returns raw knowledge-base documents."""
    from cds_textagent.rag_pipeline.retriever import vectorstore

    cid = request.case_id.strip()
    if not cid:
        raise HTTPException(status_code=422, detail="case_id must not be empty")

    # Build ChromaDB where filter
    if request.article_id.strip():
        where: dict = {"$and": [{"case_id": cid}, {"article_id": request.article_id.strip()}]}
    else:
        where = {"case_id": cid}

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(
            None,
            lambda: vectorstore._collection.get(
                where=where, include=["documents", "metadatas"]
            ),
        )
    except Exception as exc:
        logger.exception("[API] ChromaDB lookup error")
        raise HTTPException(status_code=500, detail=str(exc))

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    results = [
        CaseDocument(
            case_id=str(meta.get("case_id", "")),
            article_id=str(meta.get("article_id", "")),
            content=doc,
            metadata=dict(meta),
        )
        for doc, meta in zip(documents, metadatas)
    ]

    return CaseLookupResponse(
        query_id=cid,
        results=results,
        total_found=len(results),
    )
