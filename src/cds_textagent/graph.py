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
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from cds_textagent.state import MultiAgentState as GraphState
from cds_textagent.nodes import (
    planner_node,
    internet_search_node,
    rag_node,
    reasoning_node,
    chitchat_node,
)

logger = logging.getLogger(__name__)

# Router

def route_to_agents(state: GraphState) -> list[Send]:
    """
    Reads planner output and fans out to whichever agents are needed.
    Chitchat short-circuits the clinical pipeline entirely.
    """
    if state.get("is_chitchat"):
        return [Send("chitchat", state)]

    sends = []

    if state.get("internet_search_query"):
        sends.append(Send("internet_search_agent", state))

    if state.get("rag_query"):
        sends.append(Send("rag_agent", state))

    if not sends:
        # Fallback: if planner nulled everything, go straight to reasoning
        logger.warning("[Router] No agents selected — routing directly to reasoning.")
        sends.append(Send("reasoning_agent", state))

    return sends


# Graph Builder

memory = MemorySaver()


def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes (named after agents for clean LangSmith traces)
    graph.add_node("planner",               planner_node)
    graph.add_node("internet_search_agent",  internet_search_node)
    graph.add_node("rag_agent",              rag_node)
    graph.add_node("reasoning_agent",        reasoning_node)
    graph.add_node("chitchat",               chitchat_node)

    # START → planner
    graph.add_edge(START, "planner")

    # planner → parallel fan-out (or chitchat short-circuit)
    graph.add_conditional_edges(
        "planner",
        route_to_agents,
        ["internet_search_agent", "rag_agent", "reasoning_agent", "chitchat"],
    )

    # Both parallel agents feed into reasoning
    graph.add_edge("internet_search_agent", "reasoning_agent")
    graph.add_edge("rag_agent",             "reasoning_agent")

    # reasoning → END
    graph.add_edge("reasoning_agent", END)

    # chitchat short-circuit → END
    graph.add_edge("chitchat", END)

    return graph.compile(checkpointer=memory)


# App Instance + Public Entry Point

app = build_graph()


def ask(question: str, session_id: str = "default") -> str:
    """Run the full multiagent pipeline and return the final answer."""
    from cds_textagent.state import create_initial_state
    initial_state = create_initial_state(query=question, session_id=session_id)

    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke(initial_state, config=config)
    return result["final_answer"]


# Run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "=" * 70)
    print("  Clinical Decision Support System — Multi-Agent Pipeline")
    print("=" * 70)
    print("Ask clinical questions. The system will search guidelines,")
    print("PubMed, and the internal knowledge base, then synthesize.")
    print("Type 'exit', 'quit', or 'q' to end.")
    print("=" * 70 + "\n")

    session_id = "interactive"

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        try:
            from cds_textagent.state import create_initial_state
            initial_state = create_initial_state(query=question, session_id=session_id)
            config = {"configurable": {"thread_id": session_id}}

            print()
            final_answer = ""
            for event in app.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, update in event.items():
                    print(f"  \u2713 {node_name} completed")
                    if "final_answer" in update:
                        final_answer = update["final_answer"]

            print(f"\nAssistant:\n{'─' * 60}\n{final_answer}\n{'─' * 60}")
        except Exception as e:
            print(f"\nError: {e}")