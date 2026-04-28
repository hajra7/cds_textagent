"""
agents.py — Agent Definitions
==============================
Default backend: Mistral via Ollama (http://localhost:11434/v1).

Uses explicit tool orchestration — LLM outputs JSON tool decision, Python
executes the tool, LLM synthesizes the result. Required because most local
model quantizations do not support the OpenAI tool-calling API.

To switch backends, set env vars in your .env:
  Mistral (default) : AGENT_LLM_MODEL=mistral          AGENT_LLM_BASE_URL=http://localhost:11434/v1
  Llama             : AGENT_LLM_MODEL=llama3:8b-instruct  AGENT_LLM_BASE_URL=http://localhost:11434/v1
  GPT               : AGENT_LLM_MODEL=gpt-4o            AGENT_LLM_BASE_URL=  (leave empty)
"""

import json
import logging

from langchain_mistralai import ChatMistralAI
# from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import SecretStr

from cds_textagent.tools import (
    search_guidelines,
    search_differential_diagnosis,
    search_pubmed,
    search_knowledge_base,
)
from cds_textagent.config import AGENT_LLM_MODEL, AGENT_LLM_API_KEY

logger = logging.getLogger(__name__)

# ── Shared helpers ─────────────────────────────────────────────────────────────

_TOOL_MAP = {
    "search_guidelines": search_guidelines,
    "search_differential_diagnosis": search_differential_diagnosis,
    "search_pubmed": search_pubmed,
    "search_knowledge_base": search_knowledge_base,
}


def _make_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=AGENT_LLM_MODEL,
        mistral_api_key=SecretStr(AGENT_LLM_API_KEY) if AGENT_LLM_API_KEY else None,
        temperature=0,
    )
# def _make_llm() -> ChatOpenAI:
#     return ChatOpenAI(
#         model=AGENT_LLM_MODEL,
#         base_url=AGENT_LLM_BASE_URL,
#         api_key=SecretStr(AGENT_LLM_API_KEY) if AGENT_LLM_API_KEY else None,
#         temperature=0,
#     )


# ── OLLAMA MODE — explicit tool orchestration ──────────────────────────────────
# Used when AGENT_LLM_BASE_URL is set (local models without native tool calling).

def _run_tool(tool_name: str, query: str) -> str:
    tool = _TOOL_MAP[tool_name]
    return str(tool.invoke(query))


def _parse_tool_decision(raw: str, valid_tools: list[str], fallback: str, query: str) -> tuple[str, str]:
    """Parse LLM JSON tool decision. Returns (tool_name, tool_input)."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        decision = json.loads(raw)
        tool_name = decision["tool_name"]
        tool_input = decision.get("tool_input", query)
        if tool_name not in valid_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_name, tool_input
    except Exception:
        logger.warning("[Agent] Tool decision parse failed, using fallback. Raw: %s", raw[:120])
        return fallback, query


_IS_TOOL_PROMPT = """\
You are a clinical assistant. Choose the single most relevant tool for this question.

Available tools:
- search_guidelines             : clinical practice guidelines and protocols
- search_differential_diagnosis : differential diagnosis lists
- search_pubmed                 : research articles and clinical evidence

Return ONLY valid JSON — no markdown, no explanation:
{{"tool_name": "<search_guidelines|search_differential_diagnosis|search_pubmed>", "tool_input": "<search query>"}}

Question: {query}"""

_IS_FINAL_PROMPT = """\
You are a clinical assistant. Summarize the key findings from this search result.

Question: {query}
Tool used: {tool_name}
Result:
{tool_result}

Provide a concise, factual clinical summary."""

_RAG_FINAL_PROMPT = """\
You are a clinical assistant. Summarize relevant findings from this knowledge base result.

Question: {query}
Knowledge base result:
{tool_result}

Provide a concise, factual clinical summary based only on the retrieved content."""


class _InternetSearchAgent:
    """Internet search agent using explicit tool orchestration (Ollama mode)."""

    def __init__(self):
        self.llm = _make_llm()

    def invoke(self, inputs: dict) -> dict:
        query = inputs["messages"][0].content

        raw = str(self.llm.invoke(_IS_TOOL_PROMPT.format(query=query)).content).strip()
        tool_name, tool_input = _parse_tool_decision(
            raw,
            valid_tools=["search_guidelines", "search_differential_diagnosis", "search_pubmed"],
            fallback="search_differential_diagnosis",
            query=query,
        )

        logger.info("[InternetSearch] Tool: %s | Query: %s", tool_name, tool_input[:80])
        tool_result = _run_tool(tool_name, tool_input)

        final = self.llm.invoke(_IS_FINAL_PROMPT.format(
            query=query, tool_name=tool_name, tool_result=tool_result,
        )).content

        # Populate tool_calls so instrumentation and RAGAS can parse the trace
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {"query": tool_input}, "id": tool_name}],
        )
        return {
            "messages": [
                tool_call_msg,
                ToolMessage(content=tool_result, tool_call_id=tool_name, name=tool_name),
                AIMessage(content=str(final)),
            ]
        }


class _RAGAgent:
    """RAG agent using explicit tool orchestration (Ollama mode)."""

    def __init__(self):
        self.llm = _make_llm()

    def invoke(self, inputs: dict) -> dict:
        query = inputs["messages"][0].content

        logger.info("[RAG] Query: %s", query[:80])
        tool_result = _run_tool("search_knowledge_base", query)

        final = self.llm.invoke(_RAG_FINAL_PROMPT.format(
            query=query, tool_result=tool_result,
        )).content

        # Populate tool_calls so instrumentation and RAGAS can parse the trace
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": query}, "id": "search_knowledge_base"}],
        )
        return {
            "messages": [
                tool_call_msg,
                ToolMessage(content=tool_result, tool_call_id="search_knowledge_base", name="search_knowledge_base"),
                AIMessage(content=str(final)),
            ]
        }


class _ReasoningAgent:
    """Reasoning agent — synthesizes evidence without additional tool calls (Ollama mode)."""

    def __init__(self):
        self.llm = _make_llm()

    def invoke(self, inputs: dict) -> dict:
        prompt = inputs["messages"][0].content
        final = self.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=str(final))]}


# ── Initialise agents ──────────────────────────────────────────────────────────

logger.info("[Agents] Mistral native mode: explicit tool orchestration (%s)", AGENT_LLM_MODEL)

internet_search_agent = _InternetSearchAgent()
rag_agent             = _RAGAgent()
reasoning_agent       = _ReasoningAgent()
