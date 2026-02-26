"""
test_tools.py — Test each tool in isolation
Run: python test_tools.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.tools import search_guidelines, search_differential_diagnosis, search_pubmed


def print_result(label: str, result: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2))
    except Exception:
        print(result)


def test_search_guidelines():
    result = search_guidelines.invoke({"condition": "glioblastoma elderly"})
    print_result("search_guidelines → glioblastoma elderly", result)


def test_search_differential_diagnosis():
    result = search_differential_diagnosis.invoke({"symptoms": "headache vision loss papilledema"})
    print_result("search_differential_diagnosis → headache vision loss papilledema", result)


def test_search_pubmed():
    result = search_pubmed.invoke({"query": "glioblastoma treatment elderly patients", "max_results": 3})
    print_result("search_pubmed → glioblastoma treatment elderly", result)


if __name__ == "__main__":
    print("\n>>> Testing Tools\n")

    test_search_guidelines()
    test_search_differential_diagnosis()
    test_search_pubmed()

    print("\n>>> All tool tests done.\n")