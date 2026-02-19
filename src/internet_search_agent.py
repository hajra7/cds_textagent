from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.tools import tool_node
from langchain_community.tools import DuckDuckGoSearchResults
import json
from dotenv import load_dotenv
import os
from config import PUBMED_API_KEY

# For PubMed search
from Bio import Entrez
import time
from datetime import datetime

load_dotenv()

# PubMed setup
Entrez.email = os.getenv("PUBMED_EMAIL")


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

@tool
def search_pubmed(query: str, max_results: int = 5, years_ago: int = 10) -> str:
    """
    Search Pubmed (NCBI) directly for peer-reviewed medical literature.

    Args:
        query: Medical search query.
        max_results: Maximum number of results to return.
        years_ago: Maximum number of years to search back.

    Returns:
        JSON sring with Pubmed articles including titles, abstracts, author and PMIDs
    """
    try:
        #Build date filter
        date_filter = ""
        if years_ago > 0:
            current_year = datetime.now().year
            start_year = current_year - years_ago
            date_filter = f" AND ({start_year} : {current_year})"

            #Search Pubmed
        search_query = f"{query} {date_filter}"
        search_handle = Entrez.esearch(
            db="pubmed", 
            term=search_query, 
            retmax=max_results,
            sort="relevance")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]

        if not id_list:
            return json.dumps({
                "query": query,
                "database": "PubMed (NCBI)",
                "results_found": 0,
                "message": "No PubMed articles found for the query"
            }, indent=2)

                #Fetch article details
        time.sleep(0.34)

        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="medline",
            retmode="xml")
        articles = Entrez.read(fetch_handle)
        fetch_handle.close()

        #Parse results
        formatted_results = []
        for idx, article in enumerate(articles["PubmedArticle"], 1):
            try:
                medline = article["MedlineCitation"]["Article"]
                article_data = medline["Article"]

                #Extract information
                pmid = str(medline["PMID"])
                title = article_data.get("ArticleTitle", "N/A")

                #Abstract
                abstract_texts = article_data.get("Abstract", {}).get("AbstractText", "N/A")
                if abstract_texts:
                    abstract = "".join([str(text) for text in abstract_texts])[:500]
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
                pub_date = article_data.get("DateCompleted", medline.get("DateRevised", {}))
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
                print(f"Error processing article: {e}")
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

    



