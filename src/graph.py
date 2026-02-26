"""
graph.py — Graph Assembly
==========================
Wires nodes together and exposes a single `ask()` entry point.

Graph structure:
  START
    └─► planner_node
          └─► [conditional fan-out via route_to_agents]
                ├─► internet_search_node ─┐
                └─► rag_node             ─┤
                                          └─► reasoning_node ─► END
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from state import MultiAgentState as GraphState
from nodes import (
    planner_node,
    internet_search_node,
    rag_node,
    reasoning_node,
)

logger = logging.getLogger(__name__)

# Router

def route_to_agents(state: GraphState) -> list[Send]:
    """
    Reads planner output and fans out to whichever agents are needed.
    Agents with a null query are skipped entirely.
    """
    sends = []

    if state.get("internet_search_query"):
        sends.append(Send("internet_search_node", state))

    if state.get("rag_query"):
        sends.append(Send("rag_node", state))

    if not sends:
        # Fallback: if planner nulled everything, go straight to reasoning
        logger.warning("[Router] No agents selected — routing directly to reasoning.")
        sends.append(Send("reasoning_node", state))

    return sends


# Graph Builder

def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("planner_node",          planner_node)
    graph.add_node("internet_search_node",  internet_search_node)
    graph.add_node("rag_node",              rag_node)
    graph.add_node("reasoning_node",        reasoning_node)

    # START → planner
    graph.add_edge(START, "planner_node")

    # planner → parallel fan-out
    graph.add_conditional_edges(
        "planner_node",
        route_to_agents,
        ["internet_search_node", "rag_node", "reasoning_node"],
    )

    # Both parallel agents feed into reasoning
    graph.add_edge("internet_search_node", "reasoning_node")
    graph.add_edge("rag_node",             "reasoning_node")

    # reasoning → END
    graph.add_edge("reasoning_node", END)

    return graph.compile()


# App Instance + Public Entry Point

app = build_graph()


def ask(question: str, session_id: str = "default") -> str:
    """Run the full multiagent pipeline and return the final answer."""
    from state import create_initial_state
    initial_state = create_initial_state(query=question, session_id=session_id)

    result = app.invoke(initial_state)
    return result["final_answer"]


# Run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    answer = ask("What is the treatment for glioblastoma for elderly patients?")
    print("\n" + "=" * 60)
    print(answer)