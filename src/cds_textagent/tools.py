from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
from typing import Any
from langchain_core.tools import tool

import json
import time
from datetime import datetime
from Bio import Entrez

import logging

from cds_textagent.rag_pipeline.retriever import retrieve_and_format

logger = logging.getLogger(__name__)

load_dotenv()

Entrez.timeout = 30  # type: ignore[attr-defined]  # seconds — prevent indefinite hangs

try:
    search = DuckDuckGoSearchAPIWrapper()
except Exception:
    search = None

Entrez.email = (os.getenv("PUBMED_EMAIL") or "").strip()

#Internet Search tools
# 1. Guidelines Search Tool
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
   

# 2. Differential Diagnosis Search Tool
@tool
def search_differential_diagnosis(symptoms: str, max_results: int = 5) -> str:
    """
    Search for differential diagnosis based on symptoms. 

    Args:
        symptoms: The symptoms or signs of the patient.
        max_results: The maximum number of results to return.

    Returns:
        JSON string with potential diagnosis and recommendations.
    """
    if search is None:
        return json.dumps({"error": "Web search not available"})

    try:
        query = f"differential diagnosis for {symptoms}"
        results = search.results(query, max_results)

        formatted_results = []
        for idx, result in enumerate(results, 1):
            formatted_results.append({
                "rank": idx,
                "potential_diagnosis": result.get("title", "N/A"),
                "diagnosis_summary": result.get("snippet", "N/A"),
                "reference": result.get("link", "N/A")
            })
        
        return json.dumps({
            "symptoms_searched": symptoms,
            "potential_diagnoses": formatted_results,
            "disclaimer": "These are potential diagnoses from literature. Always consult healthcare professionals."
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

# 3. PubMed Search Tool
@tool
def search_pubmed(query: str, max_results: int = 5, years_ago: int = 10) -> str:
    """
    Search Pubmed (NCBI) directly for peer-reviewed medical literature.

    Args:
        query: Medical search query.
        max_results: Maximum number of results to return.
        years_ago: Maximum number of years to search back.

    Returns:
        JSON string with PubMed articles including titles, abstracts, authors and PMIDs
    """
    try:
        if not Entrez.email:
            return json.dumps({
                "error": "PUBMED_EMAIL is not configured.",
                "suggestion": "Set PUBMED_EMAIL in your environment or .env file."
            })

        date_filter = ""
        if years_ago > 0:
            current_year = datetime.now().year
            start_year = current_year - years_ago
            date_filter = f" AND {start_year}:{current_year}[pdat]"

        search_query = f"{query} {date_filter}"
        search_handle = Entrez.esearch(
            db="pubmed", 
            term=search_query, 
            retmax=max_results,
            sort="relevance")
        search_results: dict[str, Any] = dict(Entrez.read(search_handle))  # type: ignore[arg-type]
        search_handle.close()

        id_list = list(search_results.get("IdList", []))

        if not id_list:
            return json.dumps({
                "query": query,
                "database": "PubMed (NCBI)",
                "results_found": 0,
                "message": "No PubMed articles found for the query"
            }, indent=2)

        time.sleep(0.34)

        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="medline",
            retmode="xml")
        articles: dict[str, Any] = dict(Entrez.read(fetch_handle))  # type: ignore[arg-type]
        fetch_handle.close()

        #Parse results
        formatted_results = []
        for idx, article in enumerate(articles.get("PubmedArticle", []), 1):
            try:
                citation = article.get("MedlineCitation", {})
                article_data = citation.get("Article", {})

                #Extract information
                pmid = str(citation.get("PMID", "N/A"))
                title = article_data.get("ArticleTitle", "N/A")

                #Abstract
                abstract_texts = article_data.get("Abstract", {}).get("AbstractText")
                if isinstance(abstract_texts, (list, tuple)):
                    abstract = " ".join(str(text) for text in abstract_texts)[:500]
                elif abstract_texts:
                    abstract = str(abstract_texts)[:500]
                else:
                    abstract = "No abstract available"

                #Authors
                authors_list = article_data.get("AuthorList", [])
                authors = []
                for author in authors_list[:3]:
                    if 'LastName' in author and 'Initials' in author:
                        authors.append(f"{author['LastName']} {author['Initials']}")
                authors_str = ', '.join(authors)
                if len(authors_list) > 3:
                    authors_str += ' et al.'

                #Journal and date
                journal = article_data.get("Journal", {}).get("Title", "Unknown Journal")
                pub_date = (
                    article_data.get("Journal", {})
                    .get("JournalIssue", {})
                    .get("PubDate", {})
                )
                if not pub_date:
                    pub_date = citation.get("DateCompleted", citation.get("DateRevised", {}))
                year = pub_date.get("Year", "N/A")

                formatted_results.append({
                    "rank": idx,
                    "pmid": pmid,
                    "title": title,
                    "authors": authors_str if authors_str else "Author not specified",
                    "journal": journal,
                    "year": year,
                    "abstract": abstract,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                })

            except Exception as e:
                logger.warning("Error processing article: %s", e)
                continue

        return json.dumps({
            "query": query,
            "database": "PubMed (NCBI)",
            "results_found": len(formatted_results),
            "articles": formatted_results,
            "note": "PubMed contains peer-reviewed biomedical literature. Always review full article for complete context."
        }, indent=2)
                
    except Exception as e:
        return json.dumps({
            "error": f"PubMed Search failed: {str(e)}",
            "suggestion": "Check your internet connection and Entrez email address."
        })


# 4. Knowledge Base Search Tool (RAG)
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal clinical knowledge base for relevant case studies and articles.

    Uses embedding-based retrieval with cross-encoder reranking over the
    local ChromaDB vector store.

    Args:
        query: Clinical question or search query.

    Returns:
        Formatted string with the top matching documents, including
        embedding similarity and reranker scores.
    """
    try:
        context = retrieve_and_format(query)
        return context
    except Exception as e:
        return json.dumps({"error": f"Knowledge base search failed: {str(e)}"})


if __name__ == "__main__":
    pass

