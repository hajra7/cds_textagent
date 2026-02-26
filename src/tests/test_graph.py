"""
test_graph.py — Test the full multiagent pipeline end to end
Run: python test_graph.py
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")

from state import create_initial_state
from graph import app


def run_pipeline(query: str):
    print(f"\n{'='*60}")
    print(f"  Query: {query}")
    print(f"{'='*60}")

    initial_state = create_initial_state(query=query, session_id="test-session-001")
    result = app.invoke(initial_state)

    # ── Planner output ──────────────────────────────────────────────
    print("\n--- PLANNER ---")
    print(f"  Intent:                 {result.get('intent')}")
    print(f"  Reasoning Focus:        {result.get('reasoning_focus')}")
    print(f"  Internet Search Query:  {result.get('internet_search_query')}")
    print(f"  RAG Query:              {result.get('rag_query')}")

    # ── Agent results ───────────────────────────────────────────────
    print("\n--- AGENT RESULTS ---")
    for evidence in result.get("evidence", []):
        print(f"\n  Source: {evidence['source'].upper()}")
        print(f"  Content preview: {evidence['content'][:300]}...")
        print(f"  Citations ({len(evidence['citations'])}):")
        for c in evidence["citations"]:
            print(f"    [{c['id']}] {c['title']} — {c['url']}")

    # ── Observability ───────────────────────────────────────────────
    print("\n--- TRACE ---")
    print(f"  Agents used:     {result.get('agents_used')}")
    print(f"  Execution trace: {result.get('execution_trace')}")

    # ── Final answer ────────────────────────────────────────────────
    print("\n--- FINAL ANSWER ---")
    print(result.get("final_answer", "No answer generated."))


if __name__ == "__main__":
    print("\n>>> Testing Full Pipeline\n")

    # Test 1 — clinical question (should trigger both agents)
    run_pipeline("An 80 year old patient with a history of hypertension and diabetes presents with a sudden onset of chest pain. What is the differential diagnosis and treatment options?")

    # Test 2 — differential diagnosis question
    # run_pipeline("Patient presents with headache, vision loss, and papilledema. What are the differentials?")

    # Test 3 — guideline-only question (uncomment to test)
    # run_pipeline("What are the latest AHA guidelines for heart failure management?")

    print("\n>>> Full pipeline test done.\n")