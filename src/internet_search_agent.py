from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.tools import tool_node
from langchain_community.tools import DuckDuckGoSearchResults
import json

search = DuckDuckGoSearchResults(max_results=5)

# Tool to search for treatment guidelines
@tool
def search_guidelines(condition: str, max_results: int = 5) -> str:
    """
    Search for evidence-based guidelines for a specific medical condition.
    
    Args:
        condition: The medical condition or diagnosis.
        max_results: The maximum number of results to return.
        
    Returns:
        JSON string with treatment guidelines and recommendations.
    """
    if search is None:
        return json.dumps({"error": "Web search not available"})

    try:
        query = f"{condition} treatment guidelines and clinical practice recommendations"
        results = search.results(query, max_results)

        formatted_results = []
        for idx, result in enumerate(results, 1):
            formatted_results.append({
                "rank": idx,
                "title": result.get("title", "N/A"),
                "guideline_summary": result.get("snippet", "N/A"),
                "source_url": result.get("link", "N/A"),
            })

        return json.dumps({
            "condition": condition,
            "guidelines_found": len(formatted_results),
            "guidelines": formatted_results,
            "note": "Always verify the guidelines with official medical sources (WHO, CDC, Mayo Clinic, Apollo etc.)"
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})
   

@tool

    

