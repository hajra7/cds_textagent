# nodes.py — All Node Functions


import json
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
# from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from cds_textagent.state import MultiAgentState as GraphState
from cds_textagent.agents import internet_search_agent, rag_agent, reasoning_agent
from cds_textagent.config import PLANNER_LLM_MODEL, PLANNER_LLM_API_KEY

logger = logging.getLogger(__name__)


# LLMs

planner_llm = ChatMistralAI(
    model=PLANNER_LLM_MODEL,
    mistral_api_key=SecretStr(PLANNER_LLM_API_KEY) if PLANNER_LLM_API_KEY else None,
    temperature=0,
)
# planner_llm = ChatOpenAI(
#     model=PLANNER_LLM_MODEL,
#     base_url=PLANNER_LLM_BASE_URL,
#     api_key=SecretStr(PLANNER_LLM_API_KEY) if PLANNER_LLM_API_KEY else None,
#     temperature=0,
# )


# Node 1 — Planner

PLANNER_SYSTEM = """You are a medical query planner.
Analyze the user's input and return a JSON object:
{
  "is_chitchat": <true if the message is a greeting, small talk, or anything unrelated to medicine>,
  "intent": "<one-sentence summary of what the user needs>",
  "reasoning_focus": "<guidance for the final synthesis — what angle to prioritize>",
  "internet_search_query": "<query for guidelines, differential diagnosis, and PubMed search, or null>",
  "rag_query": "<query for internal knowledge base retrieval, or null>"
}

Rules:
- If is_chitchat is true, set both internet_search_query and rag_query to null.
- Set a query to null ONLY if that source is clearly not needed.
- For clinical questions always include both internet_search_query and rag_query.
- For general knowledge questions, rag_query may be null.
- Return ONLY valid JSON. No markdown fences."""


def planner_node(state: GraphState) -> dict:
    """Understands user intent and creates targeted queries for each agent."""
    logger.info("[Planner] Analyzing: %s", state["query"])

    # Build conversation history for follow-up awareness
    history_text = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content[:200]}...\n"

    content = state["query"]
    if history_text:
        content = f"Conversation history:\n{history_text}\n\nCurrent question: {content}"

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=content),
    ])

    try:
        raw = str(response.content).strip()
        # Strip markdown fences if the LLM wraps JSON in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        plan = json.loads(raw)
    except (json.JSONDecodeError, IndexError) as e:
        logger.error("[Planner] Failed to parse JSON: %s — raw: %s", e, response.content[:200])
        plan = {
            "intent": state["query"],
            "reasoning_focus": "answer the user query directly",
            "internet_search_query": state["query"],
            "rag_query": state["query"],
        }

    logger.info("[Planner] Plan: %s", plan)

    return {
        "route_decision":         plan.get("intent", ""),
        "is_chitchat":            bool(plan.get("is_chitchat", False)),
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
    query = state["internet_search_query"] or state["query"]
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
                tool_data = json.loads(str(msg.content))
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
        }],
        "agents_used": ["internet_search"],
        "execution_trace": [f"internet_search: {len(citations)} citation(s) retrieved"],
    }


# Node 2b — RAG Agent

def rag_node(state: GraphState) -> dict:
    """
    Runs the ReAct RAG agent.
    The agent autonomously searches the internal knowledge base
    (search_knowledge_base tool) and summarises findings.
    """
    query = state["rag_query"] or state["query"]
    logger.info("[RAG] Query: %s", query)

    result = rag_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    final_text = ""
    citations: list[dict] = []

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            case_ids = re.findall(r"\[Doc \d+ \| CaseID: ([^\|]+?)\s*\|", str(msg.content))
            if case_ids:
                for case_id in case_ids:
                    cid = case_id.strip()
                    citations.append({
                        "id":      f"rag-{len(citations) + 1}",
                        "title":   "Internal Knowledge Base",
                        "case_id": cid,
                        "url":     cid,
                        "source":  "rag",
                    })
            else:
                citations.append({
                    "id":      f"rag-{len(citations) + 1}",
                    "title":   "Internal Knowledge Base",
                    "case_id": "unknown",
                    "url":     "",
                    "source":  "rag",
                })

        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content

    return {
        "evidence": [{
            "source":    "rag",
            "content":   final_text,
            "citations": citations,
        }],
        "agents_used": ["rag"],
        "execution_trace": [f"rag: {len(citations)} chunk(s) retrieved"],
    }


# Node 3 — Reasoning Agent

def reasoning_node(state: GraphState) -> dict:
    """
    Runs the ReAct reasoning agent.
    The agent receives all gathered evidence, synthesises it, and may
    call additional tools (search_knowledge_base, search_pubmed,
    search_guidelines) to verify or fill gaps.
    """
    logger.info("[Reasoning] Synthesizing %d result(s)", len(state["evidence"]))

    evidence_blocks = []
    all_citations: list[dict] = []

    for result in state["evidence"]:
        if result["content"]:
            evidence_blocks.append(
                f"### {result['source'].upper().replace('_', ' ')} EVIDENCE\n{result['content']}"
            )
        all_citations.extend(result["citations"])

    evidence_text = "\n\n".join(evidence_blocks) or "No evidence retrieved."
    citations_block = json.dumps(all_citations, indent=2)

    prompt = f"""You are a senior clinical expert synthesizing evidence from multiple sources.

Instructions:
- Cite every factual claim inline using [citation_id] format
- Structure your response as:
  **Clinical Summary** — 2-3 sentence overview
  **Evidence from Literature** — findings from PubMed and guidelines
  **Internal Knowledge Base** — relevant context from RAG retrieval
  **Clinical Recommendations** — actionable guidance
  **Conclusion**
- End with a **References** section listing all cited sources.
  For internet/PubMed citations use: [id] Title. Source. [URL](URL)
  For internal knowledge base citations (source: rag) use: [id] Internal Knowledge Base. Case ID: <case_id value from citation>
- Acknowledge conflicting evidence where present
- Omit sections gracefully if a source returned no useful content
- You have tools available — use them ONLY if the existing evidence has clear gaps

User Question: {state["query"]}

Planner Intent: {state["route_decision"]}
Synthesis Focus: {state["reasoning_focus"]}

--- RETRIEVED EVIDENCE ---
{evidence_text}

--- AVAILABLE CITATIONS ---
{citations_block}

Write a comprehensive, well-structured clinical answer using [citation_id] inline citations."""

    result = reasoning_agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    # Last AIMessage is the final synthesis
    final_text = ""
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content

    return {
        "final_answer": final_text,
        "agents_used": ["reasoning"],
        "execution_trace": ["reasoning: synthesis complete"],
        "messages": [
            HumanMessage(content=state["query"]),
            AIMessage(content=final_text),
        ],
    }


# Node 4 — Chitchat

CHITCHAT_SYSTEM = """You are a professional clinical decision support assistant.
- For greetings: respond briefly and professionally, and invite the user to ask a clinical or medical question.
- For off-topic requests: politely dismiss the topic, state that you are limited to clinical and medical queries, and encourage the user to ask a medical question instead.
Keep all responses concise — no more than 2 sentences."""


def chitchat_node(state: GraphState) -> dict:
    """Returns a friendly response for greetings and off-topic messages."""
    logger.info("[Chitchat] Handling general message: %s", state["query"])

    response = planner_llm.invoke([
        SystemMessage(content=CHITCHAT_SYSTEM),
        HumanMessage(content=state["query"]),
    ])
    answer = str(response.content).strip()

    return {
        "final_answer": answer,
        "agents_used": ["chitchat"],
        "execution_trace": ["chitchat: responded to general message"],
        "messages": [
            HumanMessage(content=state["query"]),
            AIMessage(content=answer),
        ],
    }