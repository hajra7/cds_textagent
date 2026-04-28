"""
retriever.py — Multi-Strategy Retrieval
========================================
Supports four retrieval strategies:
  - dense:       ChromaDB embedding similarity + cross-encoder rerank
  - hybrid:      Dense + BM25 fused via Reciprocal Rank Fusion + rerank
  - ensemble:    Dense + BM25 + keyword-augmented dense via weighted RRF + rerank
  - multi_query: LLM generates query reformulations, retrieves for each, deduplicates + rerank
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
# from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from cds_textagent import config
from cds_textagent.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────

_PERSIST_DIR = PROCESSED_DATA_DIR / "chroma_openai_text_embedding_3_small"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=str(_PERSIST_DIR),
    embedding_function=embeddings,
    collection_name="cases_openai",
)

os.environ["TQDM_DISABLE"] = "1"
cross_encoder = CrossEncoder(
    _RERANKER_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
os.environ.pop("TQDM_DISABLE")


# ── Shared helpers ───────────────────────────────────────────────

def _rerank(question: str, docs_with_similarity, top_k: int = 5):
    """Cross-encoder rerank over (Document, similarity) pairs."""
    if not docs_with_similarity:
        return []

    pairs = [(question, doc.page_content[:1200]) for doc, _sim in docs_with_similarity]
    scores = cross_encoder.predict(pairs)

    reranked = [
        (doc, sim, float(ce_score))
        for (doc, sim), ce_score in zip(docs_with_similarity, scores)
    ]
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked[:top_k]


def _to_similarity(score, space: str) -> float:
    """Convert Chroma distance to similarity (0-1)."""
    if space == "cosine":
        sim = 1.0 - float(score)
    else:
        sim = float(score)
    return max(0.0, min(1.0, sim))


def _format_reranked_docs(top_reranked: List[Tuple[Document, float, float]]) -> str:
    """Format reranked (doc, embedding_sim, ce_score) triples into display string."""
    if not top_reranked:
        return "No relevant context found."
    formatted_chunks = []
    for i, (doc, sim, ce) in enumerate(top_reranked, 1):
        case_id = (
            doc.metadata.get("case_id")
            or doc.metadata.get("article_id")
            or f"row-{doc.metadata.get('row_id', 'unknown')}"
        )
        formatted_chunks.append(
            f"[Doc {i} | CaseID: {case_id} | EmbeddingSim: {sim:.4f} ({sim*100:.1f}%) | RerankScore: {ce:.4f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def _get_chroma_space() -> str:
    """Get the HNSW distance space from ChromaDB collection metadata."""
    try:
        collection = vectorstore._collection
        return (collection.metadata or {}).get("hnsw:space", "cosine")
    except AttributeError:
        return "cosine"


def _dense_retrieve(question: str, k: int = 30, min_similarity: float | None = None):
    """Run dense retrieval via ChromaDB. Returns [(Document, similarity)]."""
    space = _get_chroma_space()
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)
    docs_with_similarity = [
        (doc, _to_similarity(score, space)) for doc, score in docs_with_scores
    ]
    docs_with_similarity.sort(key=lambda x: x[1], reverse=True)
    if min_similarity is not None:
        docs_with_similarity = [(d, s) for d, s in docs_with_similarity if s >= min_similarity]
    return docs_with_similarity


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    weights: List[float] | None = None,
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Fuse multiple ranked lists using weighted Reciprocal Rank Fusion.

    Each inner list is [(Document, score)] sorted descending by score.
    Documents are matched across lists by doc.metadata["doc_id"].
    Returns fused list of (Document, rrf_score) sorted descending.
    """
    if not ranked_lists:
        return []

    n_lists = len(ranked_lists)
    if weights is None:
        weights = [1.0 / n_lists] * n_lists

    # Build rank maps: doc_id -> rank (0-indexed)
    rank_maps: List[Dict[str, int]] = []
    doc_store: Dict[str, Document] = {}
    for ranked_list in ranked_lists:
        rank_map = {}
        for rank, (doc, _score) in enumerate(ranked_list):
            doc_id = doc.metadata.get("doc_id", doc.page_content[:50])
            rank_map[doc_id] = rank
            doc_store[doc_id] = doc
        rank_maps.append(rank_map)

    # Compute RRF scores
    all_doc_ids = set()
    for rm in rank_maps:
        all_doc_ids.update(rm.keys())

    rrf_scores: Dict[str, float] = {}
    for doc_id in all_doc_ids:
        score = 0.0
        for i, rank_map in enumerate(rank_maps):
            if doc_id in rank_map:
                score += weights[i] * (1.0 / (k + rank_map[doc_id] + 1))
            # Documents not in a list get no contribution from that list
        rrf_scores[doc_id] = score

    # Sort by RRF score and return
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(doc_store[did], rrf_scores[did]) for did in sorted_ids]


# ── Strategy 1: Dense ────────────────────────────────────────────

def _retrieve_dense(
    question: str,
    retrieve_k: int = 30,
    final_k: int = 5,
    min_similarity: float | None = None,
) -> Tuple[str, List[Tuple[Document, float, float]]]:
    """Current dense-only strategy: ChromaDB + cross-encoder rerank."""
    docs_with_similarity = _dense_retrieve(question, k=retrieve_k, min_similarity=min_similarity)
    if not docs_with_similarity:
        return "No relevant context found.", []
    top_reranked = _rerank(question, docs_with_similarity, top_k=final_k)
    return _format_reranked_docs(top_reranked), top_reranked


# ── Strategy 2: Hybrid (Dense + BM25) ────────────────────────────

def _retrieve_hybrid(
    question: str,
    retrieve_k: int = 50,
    final_k: int = 10,
    min_similarity: float | None = None,
) -> Tuple[str, List[Tuple[Document, float, float]]]:
    """Dense + BM25 fused via Reciprocal Rank Fusion, then cross-encoder rerank."""
    from cds_textagent.rag_pipeline.bm25_index import get_bm25_index, _tokenize

    # Dense retrieval
    dense_docs = _dense_retrieve(question, k=retrieve_k, min_similarity=min_similarity)

    # BM25 retrieval
    bm25, bm25_docs = get_bm25_index()
    query_tokens = _tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)

    # Pair BM25 scores with documents, sort descending, take top retrieve_k
    bm25_ranked = sorted(
        [(bm25_docs[i], float(bm25_scores[i])) for i in range(len(bm25_docs))],
        key=lambda x: x[1],
        reverse=True,
    )[:retrieve_k]

    # Fuse via RRF
    fused = _reciprocal_rank_fusion(
        [dense_docs, bm25_ranked],
        weights=[config.DENSE_WEIGHT, config.SPARSE_WEIGHT],
        k=config.RRF_K,
    )

    if not fused:
        return "No relevant context found.", []

    # Take top candidates, rerank
    candidates = [(doc, score) for doc, score in fused[:retrieve_k]]
    top_reranked = _rerank(question, candidates, top_k=final_k)
    return _format_reranked_docs(top_reranked), top_reranked


# ── Strategy 3: Ensemble (Dense + BM25 + Keyword-augmented Dense) ─

def _retrieve_ensemble(
    question: str,
    retrieve_k: int = 50,
    final_k: int = 10,
    min_similarity: float | None = None,
) -> Tuple[str, List[Tuple[Document, float, float]]]:
    """
    Three sub-retrievers fused via weighted RRF:
      1. Dense (ChromaDB)
      2. BM25 (sparse)
      3. Keyword-augmented dense (keywords appended to query)
    """
    from cds_textagent.rag_pipeline.bm25_index import get_bm25_index, _tokenize
    from cds_textagent.rag_pipeline.text_utils import extract_keywords

    # Sub-retriever 1: Dense
    dense_docs = _dense_retrieve(question, k=retrieve_k, min_similarity=min_similarity)

    # Sub-retriever 2: BM25
    bm25, bm25_docs = get_bm25_index()
    query_tokens = _tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked = sorted(
        [(bm25_docs[i], float(bm25_scores[i])) for i in range(len(bm25_docs))],
        key=lambda x: x[1],
        reverse=True,
    )[:retrieve_k]

    # Sub-retriever 3: Keyword-augmented dense
    keywords = extract_keywords(question, max_keywords=5)
    augmented_query = f"{question} {' '.join(keywords)}"
    keyword_docs = _dense_retrieve(augmented_query, k=retrieve_k, min_similarity=min_similarity)

    # Fuse via weighted RRF
    fused = _reciprocal_rank_fusion(
        [dense_docs, bm25_ranked, keyword_docs],
        weights=[0.4, 0.3, 0.3],
        k=config.RRF_K,
    )

    if not fused:
        return "No relevant context found.", []

    candidates = [(doc, score) for doc, score in fused[:retrieve_k]]
    top_reranked = _rerank(question, candidates, top_k=final_k)
    return _format_reranked_docs(top_reranked), top_reranked


# ── Strategy 4: Multi-Query ──────────────────────────────────────

def _generate_query_reformulations(
    question: str,
    num_reformulations: int = 3,
) -> List[str]:
    """Use LLM to generate diverse reformulations of a clinical query."""
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = (
        f"You are a clinical query reformulation expert. Given the original clinical "
        f"question, generate {num_reformulations} alternative formulations that:\n"
        f"- Use different medical terminology (synonyms, related terms)\n"
        f"- Approach the question from different clinical angles\n"
        f"- Maintain the same clinical intent\n\n"
        f"Return ONLY a JSON array of strings. No explanation.\n\n"
        f"Original question: {question}"
    )
    import json
    import re
    response = llm.invoke(prompt)
    text = str(response.content).strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        reformulations = json.loads(text)
        if isinstance(reformulations, list):
            return [str(r) for r in reformulations[:num_reformulations]]
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse query reformulations, using original query only")
    return []


def _retrieve_multi_query(
    question: str,
    retrieve_k: int = 30,
    final_k: int = 10,
    min_similarity: float | None = None,
) -> Tuple[str, List[Tuple[Document, float, float]]]:
    """
    Generate query reformulations, retrieve for each, deduplicate, then rerank.
    Reranking always uses the original question for relevance.
    """
    reformulations = _generate_query_reformulations(
        question, num_reformulations=config.MULTI_QUERY_NUM_REFORMULATIONS
    )
    all_queries = [question] + reformulations
    per_query_k = max(5, retrieve_k // len(all_queries))

    # Retrieve for each query
    all_docs: Dict[str, Tuple[Document, float]] = {}  # doc_id -> (doc, best_similarity)
    for q in all_queries:
        docs = _dense_retrieve(q, k=per_query_k, min_similarity=min_similarity)
        for doc, sim in docs:
            doc_id = doc.metadata.get("doc_id", doc.page_content[:50])
            if doc_id not in all_docs or sim > all_docs[doc_id][1]:
                all_docs[doc_id] = (doc, sim)

    if not all_docs:
        return "No relevant context found.", []

    # Deduplicated candidates sorted by best similarity
    candidates = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)

    # Rerank with ORIGINAL question
    top_reranked = _rerank(question, candidates, top_k=final_k)
    return _format_reranked_docs(top_reranked), top_reranked


# ── Strategy dispatcher ──────────────────────────────────────────

_STRATEGY_MAP = {
    "dense": _retrieve_dense,
    "hybrid": _retrieve_hybrid,
    "ensemble": _retrieve_ensemble,
    "multi_query": _retrieve_multi_query,
}


def retrieve_rerank_and_format_docs(
    question: str,
    retrieve_k: int = 30,
    final_k: int = 5,
    min_similarity: float | None = None,
    strategy: str | None = None,
) -> Tuple[str, List[Tuple[Document, float, float]]]:
    """
    Main entry point. Dispatches to the configured retrieval strategy.

    Args:
        question: The query string.
        retrieve_k: Number of initial candidates to retrieve.
        final_k: Number of final documents after reranking.
        min_similarity: Optional minimum similarity threshold.
        strategy: Override strategy. If None, reads from config.RETRIEVAL_STRATEGY.

    Returns:
        (formatted_context_string, reranked_docs_with_scores)
    """
    if strategy is None:
        strategy = config.RETRIEVAL_STRATEGY

    retriever_fn = _STRATEGY_MAP.get(strategy)
    if retriever_fn is None:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy!r}. "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )

    return retriever_fn(
        question,
        retrieve_k=retrieve_k,
        final_k=final_k,
        min_similarity=min_similarity,
    )


def retrieve_and_format(question: str, strategy: str | None = None) -> str:
    """Single entry-point for the agent tool — returns context as a plain string."""
    context, _reranked_docs = retrieve_rerank_and_format_docs(
        question,
        retrieve_k=config.TOP_K_INITIAL,
        final_k=config.TOP_K_FINAL,
        min_similarity=config.SIMILARITY_THRESHOLD,
        strategy=strategy,
    )
    return context
