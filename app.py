"""
app.py — Streamlit frontend for the cds_textagent CDSS
=======================================================
Run with:  streamlit run app.py --server.port 8501
Requires:  api.py running on port 8000
"""

import json
import requests
import streamlit as st
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"

AGENT_COLORS = {
    "internet_search": "blue",
    "rag": "green",
    "reasoning": "orange",
    "chitchat": "gray",
}

# ---------------------------------------------------------------------------
# Episodic memory helpers
# ---------------------------------------------------------------------------

HISTORY_DIR = Path("session_history")
HISTORY_DIR.mkdir(exist_ok=True)
COUNTER_FILE = HISTORY_DIR / ".counter"


def next_session_id() -> str:
    n = int(COUNTER_FILE.read_text().strip()) + 1 if COUNTER_FILE.exists() else 1
    COUNTER_FILE.write_text(str(n))
    return f"{n:03d}"


def load_session(session_id: str) -> list:
    p = HISTORY_DIR / f"{session_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_session(session_id: str, messages: list) -> None:
    (HISTORY_DIR / f"{session_id}.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def delete_session(session_id: str) -> None:
    p = HISTORY_DIR / f"{session_id}.json"
    if p.exists():
        p.unlink()


def list_sessions() -> list[str]:
    """Return all saved session IDs, newest first."""
    return sorted(
        (f.stem for f in HISTORY_DIR.glob("[0-9]*.json")),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = next_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = load_session(st.session_state.session_id)
if "uploaded_doc" not in st.session_state:
    st.session_state.uploaded_doc = None  # {"name": str, "text": str}
if "submitted_doc" not in st.session_state:
    st.session_state.submitted_doc = None  # name of last submitted file

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("CDSS")
    st.caption(f"Active session: **{st.session_state.session_id}**")

    if st.button("＋ New Session", use_container_width=True):
        st.session_state.session_id = next_session_id()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Sessions**")

    all_sessions = list_sessions()
    if not all_sessions:
        st.caption("No saved sessions yet.")
    else:
        for sid in all_sessions:
            msgs = load_session(sid)
            last_q = next(
                (m["content"] for m in reversed(msgs) if m["role"] == "user"), None
            )
            is_active = sid == st.session_state.session_id
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    f"Session {sid}",
                    key=f"sess_{sid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        st.session_state.session_id = sid
                        st.session_state.messages = msgs
                        st.rerun()
            with col_del:
                if st.button("✕", key=f"del_{sid}", help="Delete session"):
                    delete_session(sid)
                    if is_active:
                        st.session_state.session_id = next_session_id()
                        st.session_state.messages = []
                    st.rerun()
            if last_q:
                st.caption(
                    f"_{last_q[:70]}{'…' if len(last_q) > 70 else ''}_"
                )

    st.divider()

    if st.session_state.messages:
        st.download_button(
            "Export Session (JSON)",
            data=json.dumps(st.session_state.messages, indent=2, ensure_ascii=False),
            file_name=f"cdss_session_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.caption(
        "For informational purposes only. "
        "Always consult a qualified healthcare professional."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_agent_expander(agents: list, elapsed: float, key: str) -> None:
    if not agents:
        return
    with st.expander("How was this answered?", expanded=False):
        badges = "  ".join(
            f":{AGENT_COLORS.get(a, 'gray')}[{a.replace('_', ' ').title()}]"
            for a in agents
        )
        st.caption(f"Agents: {badges}")
        if elapsed:
            st.caption(f"Response time: {elapsed:.1f}s")


def render_assistant_message(msg: dict, key_suffix: str) -> None:
    tab_summary, tab_sources = st.tabs(["Summary", "Sources"])
    with tab_summary:
        st.markdown(msg["content"])
        render_agent_expander(
            msg.get("agents_used", []), msg.get("elapsed", 0.0), key=f"exp_{key_suffix}"
        )
        st.download_button(
            "Export answer (.md)",
            data=msg["content"],
            file_name=f"cdss_{st.session_state.session_id}_{key_suffix}.md",
            mime="text/markdown",
            key=f"dl_{key_suffix}",
        )
    with tab_sources:
        if "**References**" in msg["content"]:
            refs = msg["content"].split("**References**", 1)[1]
        else:
            refs = "_No References section found in this response._"
        st.markdown(refs)


# ---------------------------------------------------------------------------
# Main tabs: Chat | Case Lookup
# ---------------------------------------------------------------------------

st.title("Clinical Decision Support System")
st.caption("Powered by Mistral LLM · ChromaDB RAG · PubMed")

chat_tab, lookup_tab = st.tabs(["Chat", "Case Lookup"])

# ── Chat tab ────────────────────────────────────────────────────────────────
with chat_tab:
    from cds_textagent.document_utils import extract_text

    # Render existing conversation
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg, key_suffix=str(i))
            else:
                st.markdown(msg["content"])

    st.divider()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a clinical document (PDF or DOCX)",
        type=["pdf", "docx", "doc"],
        help="Upload a PDF or DOCX file to analyse with the CDSS pipeline.",
    )

    if uploaded_file is not None:
        if (
            st.session_state.uploaded_doc is None
            or st.session_state.uploaded_doc["name"] != uploaded_file.name
        ):
            text, err = extract_text(uploaded_file)
            if err and err != "truncated":
                st.error(err)
                st.session_state.uploaded_doc = None
            else:
                st.session_state.uploaded_doc = {"name": uploaded_file.name, "text": text}
                st.session_state.submitted_doc = None  # reset so new file can be submitted
                if err == "truncated":
                    st.warning(f"**{uploaded_file.name}** loaded (truncated to 12,000 characters).")

    doc = st.session_state.uploaded_doc
    already_submitted = doc and st.session_state.submitted_doc == doc["name"]

    if doc:
        st.info(f"📄 **{doc['name']}** — {len(doc['text']):,} chars")

    analyse_clicked = st.button(
        "Analyse Document",
        type="primary",
        disabled=bool(not doc or already_submitted),
        use_container_width=True,
    )

    if already_submitted:
        st.caption("Document already analysed. Upload a new file to run again.")

    if analyse_clicked and doc:
        ts = datetime.now(timezone.utc).isoformat()
        user_label = f"📄 {doc['name']}"
        st.session_state.messages.append({"role": "user", "content": user_label, "timestamp": ts})
        st.session_state.submitted_doc = doc["name"]

        with st.chat_message("user"):
            st.markdown(user_label)

        question = f"Document: {doc['name']}\n{doc['text']}"

        with st.chat_message("assistant"):
            with st.spinner("Consulting clinical agents… (10–30s)"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "question": question,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data["answer"]
                    agents_used = data.get("agents_used", [])
                    elapsed = data.get("elapsed_seconds", 0.0)
                except requests.exceptions.ConnectionError:
                    answer = (
                        "Could not connect to the CDSS API. "
                        "Please ensure `api.py` is running on port 8000 (`uvicorn api:app --port 8000`)."
                    )
                    agents_used, elapsed = [], 0.0
                except requests.exceptions.Timeout:
                    answer = (
                        "The clinical reasoning pipeline timed out (>120s). "
                        "Please try a more specific question."
                    )
                    agents_used, elapsed = [], 0.0
                except Exception as exc:
                    answer = f"Unexpected error: {exc}"
                    agents_used, elapsed = [], 0.0

            reply: dict = {
                "role": "assistant",
                "content": answer,
                "agents_used": agents_used,
                "elapsed": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            render_assistant_message(reply, key_suffix=f"new_{ts}")

        st.session_state.messages.append(reply)
        save_session(st.session_state.session_id, st.session_state.messages)

    # Text input — works with or without an attached document
    if prompt := st.chat_input("Ask a clinical question…"):
        ts = datetime.now(timezone.utc).isoformat()
        doc = st.session_state.uploaded_doc
        user_label = f"📄 {doc['name']} — {prompt}" if doc else prompt
        st.session_state.messages.append({"role": "user", "content": user_label, "timestamp": ts})

        with st.chat_message("user"):
            st.markdown(user_label)

        question = (
            f"Document: {doc['name']}\n{doc['text']}\n\nQuestion: {prompt}" if doc else prompt
        )

        with st.chat_message("assistant"):
            with st.spinner("Consulting clinical agents… (10–30s)"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"question": question, "session_id": st.session_state.session_id},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data["answer"]
                    agents_used = data.get("agents_used", [])
                    elapsed = data.get("elapsed_seconds", 0.0)
                except requests.exceptions.ConnectionError:
                    answer = (
                        "Could not connect to the CDSS API. "
                        "Please ensure `api.py` is running on port 8000 (`uvicorn api:app --port 8000`)."
                    )
                    agents_used, elapsed = [], 0.0
                except requests.exceptions.Timeout:
                    answer = "The clinical reasoning pipeline timed out (>120s). Please try a more specific question."
                    agents_used, elapsed = [], 0.0
                except Exception as exc:
                    answer = f"Unexpected error: {exc}"
                    agents_used, elapsed = [], 0.0

            reply: dict = {
                "role": "assistant",
                "content": answer,
                "agents_used": agents_used,
                "elapsed": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            render_assistant_message(reply, key_suffix=f"txt_{ts}")

        st.session_state.messages.append(reply)
        save_session(st.session_state.session_id, st.session_state.messages)


# ── Case Lookup tab ─────────────────────────────────────────────────────────
with lookup_tab:
    st.subheader("Search Knowledge Base by Case ID")
    st.caption(
        "Look up specific radiology cases directly from the internal ChromaDB vector store. "
        "This does not run the full CDSS pipeline."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        case_id_input = st.text_input("Case ID", placeholder="e.g. RAD-001")
    with col2:
        article_id_input = st.text_input("Article ID (optional)", placeholder="e.g. ART-123")

    if st.button("Search", type="primary"):
        if not case_id_input.strip():
            st.warning("Please enter a Case ID.")
        else:
            with st.spinner("Querying knowledge base…"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/lookup-case",
                        json={
                            "case_id": case_id_input.strip(),
                            "article_id": article_id_input.strip(),
                        },
                        timeout=15,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    lookup_error = None
                except requests.exceptions.ConnectionError:
                    data = None
                    lookup_error = "Could not connect to the CDSS API. Ensure `api.py` is running on port 8000."
                except Exception as exc:
                    data = None
                    lookup_error = f"Lookup failed: {exc}"

            if lookup_error:
                st.error(lookup_error)
            elif data is not None:
                n = data["total_found"]
                if n == 0:
                    st.info(f"No documents found for Case ID `{data['query_id']}`.")
                else:
                    st.success(f"Found {n} document(s) for `{data['query_id']}`.")
                    for i, doc in enumerate(data["results"], 1):
                        label = f"Document {i} — Case: {doc['case_id']}"
                        if doc["article_id"]:
                            label += f" | Article: {doc['article_id']}"
                        with st.expander(label, expanded=(i == 1)):
                            st.markdown(doc["content"])
                            st.divider()
                            st.caption("Metadata")
                            st.json(doc["metadata"])
