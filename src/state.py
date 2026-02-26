# state.py — Shared Graph State
import operator
from typing import Annotated, TypedDict, Any, Dict, List, Optional


class MultiAgentState(TypedDict):
    #  Input 
    query: str
    session_id: str

    #  Planner outputs 
    route_decision: str
    clinical_struct: Dict[str, Any]
    internet_search_query: Optional[str]    # targeted query for internet agent
    rag_query: Optional[str]                # targeted query for RAG agent
    reasoning_focus: str                    # synthesis guidance for reasoning agent

    #  Parallel agent results 
    # operator.add lets parallel nodes safely append without overwriting
    evidence: Annotated[List[Dict[str, Any]], operator.add]

    #  Reasoning 
    reasoning_context: str
    reasoning_complete: bool

    #  Output 
    final_answer: str

    #  Observability 
    agents_used: Annotated[List[str], operator.add]         
    execution_trace: Annotated[List[str], operator.add]     


def create_initial_state(query: str, session_id: str) -> MultiAgentState:
    """Create the initial state for the multi-agent system."""
    return {
        # Input
        "query":                  query,
        "session_id":             session_id,

        # Planner
        "route_decision":         "",
        "clinical_struct":        {},
        "internet_search_query":  None,
        "rag_query":              None,
        "reasoning_focus":        "",

        # Parallel results
        "evidence":               [],

        # Reasoning
        "reasoning_context":      "",
        "reasoning_complete":     False,

        # Output
        "final_answer":           "",

        # Observability
        "agents_used":            [],
        "execution_trace":        [],
    }