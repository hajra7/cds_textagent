# nodes.py — All Node Functions


import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from state import MultiAgentState as GraphState
from agents import internet_search_agent

# from rag_pipeline.retriever import retrieve_and_format  # uncomment when RAG is ready

logger = logging.getLogger(__name__)


# LLMs

planner_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reasoning_llm  = ChatOpenAI(model="gpt-4o",      temperature=0)


# Node 1 — Planner

PLANNER_SYSTEM = """You are a medical query planner.
Analyze the user's medical question and return a JSON object:
{
  "intent": "<one-sentence summary of what the user needs>",
  "reasoning_focus": "<guidance for the final synthesis — what angle to prioritize>",
  "internet_search_query": "<query for guidelines, differential diagnosis, and PubMed search, or null>",
  "rag_query": "<query for internal knowledge base retrieval, or null>"
}

Rules:
- Set a query to null ONLY if that source is clearly not needed.
- For clinical questions always include both internet_search_query and rag_query.
- For general knowledge questions, rag_query may be null.
- Return ONLY valid JSON. No markdown fences."""


def planner_node(state: GraphState) -> dict:
    """Understands user intent and creates targeted queries for each agent."""
    logger.info("[Planner] Analyzing: %s", state["query"])

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=state["query"]),
    ])

    plan = json.loads(response.content)
    logger.info("[Planner] Plan: %s", plan)

    return {
        "route_decision":         plan.get("intent", ""),
        "reasoning_focus":        plan.get("reasoning_focus", ""),
        "internet_search_query":  plan.get("internet_search_query"),
        "rag_query":              plan.get("rag_query"),
        "evidence":               [],
    }


# Node 2a — Internet Search Agent

def internet_search_node(state: GraphState) -> dict:
    """
    Runs the ReAct internet search agent.
    The agent autonomously decides which tools to call
    (search_guidelines, search_differential_diagnosis, search_pubmed).
    """
    query = state["internet_search_query"]
    logger.info("[InternetSearch] Query: %s", query)

    result = internet_search_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    final_text = ""
    citations  = []

    for msg in result["messages"]:
        # Extract structured citations from every tool response
        if isinstance(msg, ToolMessage):
            try:
                tool_data = json.loads(msg.content)
                # Map each tool's JSON key to a citation list
                source_key = next(
                    (k for k in ["guidelines", "articles", "potential_diagnoses"]
                     if k in tool_data),
                    None,
                )
                if source_key:
                    for item in tool_data[source_key]:
                        citations.append({
                            "id":     f"is-{len(citations) + 1}",
                            "title":  item.get("title") or item.get("potential_diagnosis", "N/A"),
                            "url":    item.get("source_url") or item.get("reference") or item.get("pubmed_url", ""),
                            "source": "internet_search",
                        })
            except (json.JSONDecodeError, TypeError):
                pass

        # Last AIMessage with content = final agent summary
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content

    return {
        "evidence": [{
            "source":    "internet_search",
            "content":   final_text,
            "citations": citations,
        }]
    }


# Node 2b — RAG Agent

def rag_node(state: GraphState) -> dict:
    """
    Retrieves context from the internal knowledge base.

    To activate: uncomment the retrieve_and_format import at the top
    and replace the stub block below.
    """
    query = state["rag_query"]
    logger.info("[RAG] Query: %s", query)

    # ── REPLACE with real retrieval ──────────────────────────────────────
    # formatted_docs, raw_citations = retrieve_and_format(query)
    # return {
    #     "agent_results": [{
    #         "source": "rag",
    #         "content": formatted_docs,
    #         "citations": [
    #             {
    #                 "id":     f"rag-{i}",
    #                 "title":  c["title"],
    #                 "url":    c["source"],
    #                 "source": "rag",
    #             }
    #             for i, c in enumerate(raw_citations, 1)
    #         ],
    #     }]
    # }
    # ────────────────────────────────────────────────────────────────────

    return {
        "evidence": [{
            "source":    "rag",
            "content":   f"[RAG stub] No internal documents retrieved yet for: {query}",
            "citations": [],
        }]
    }


# Node 3 — Reasoning Agent

REASONING_SYSTEM = """You are a senior clinical expert synthesizing evidence from multiple sources.

Instructions:
- Cite every factual claim inline using [citation_id] format
- Structure your response as:
  **Clinical Summary** — 2-3 sentence overview
  **Evidence from Literature** — findings from PubMed and guidelines
  **Internal Knowledge Base** — relevant context from RAG retrieval
  **Clinical Recommendations** — actionable guidance
  **Conclusion**
- End with a **References** section listing all cited sources
- Acknowledge conflicting evidence where present
- Omit sections gracefully if a source returned no useful content"""


def reasoning_node(state: GraphState) -> dict:
    """Synthesizes all agent results into a final cited answer."""
    logger.info("[Reasoning] Synthesizing %d result(s)", len(state["evidence"]))

    evidence_blocks = []
    all_citations   = []

    for result in state["evidence"]:
        if result["content"]:
            evidence_blocks.append(
                f"### {result['source'].upper().replace('_', ' ')} EVIDENCE\n{result['content']}"
            )
        all_citations.extend(result["citations"])

    evidence_text   = "\n\n".join(evidence_blocks) or "No evidence retrieved."
    citations_block = json.dumps(all_citations, indent=2)

    user_prompt = f"""User Question: {state["query"]}

Planner Intent: {state["route_decision"]}
Synthesis Focus: {state["reasoning_focus"]}

--- RETRIEVED EVIDENCE ---
{evidence_text}

--- AVAILABLE CITATIONS ---
{citations_block}

Write a comprehensive, well-structured clinical answer using [citation_id] inline citations."""

    response = reasoning_llm.invoke([
        SystemMessage(content=REASONING_SYSTEM),
        HumanMessage(content=user_prompt),
    ])

    return {"final_answer": response.content}