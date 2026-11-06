# CDS-TextAgent: A Multi-Agent System for Clinically Grounded Decision Support
 
A LangGraph multi-agent **Clinical Decision Support System (CDSS)** that answers clinical questions with citation-grounded evidence. The system pulls from live medical literature (PubMed, web guidelines) and from a radiology case knowledge base.
 
This repository contains the implementation for my Master's thesis, *"Text-Based Multi-Agent System for Clinically Grounded Decision Support"* (Inje University, Department of Computer Engineering, 2026; advisor Prof. Hee-Cheol Kim).
 
>  **Research prototype only.** This is not a medical device and must not be used for real patient care.

---
 
## Why
 
LLMs can reason about clinical questions, but they hallucinate and their knowledge stops at a training cutoff. In medicine, both problems are safety risks. This project grounds every answer in retrieved evidence, attaches a citation to every claim, and keeps the LLM backbone interchangeable so different models can be compared under identical conditions.
 
## Architecture
 
```
START
  └─► Planner  (LLM → structured JSON: route, sub-queries, chitchat flag)
        ├─► Internet Search Agent ─┐   DuckDuckGo guidelines + NCBI Entrez/PubMed
        ├─► RAG Agent ─────────────┤   ChromaDB hybrid retrieval + cross-encoder rerank
        │                          └─► Reasoning Agent ─► cited clinical answer ─► END
        └─► Chitchat node ─────────────────────────────────────────────────────────► END
```

- **Planner**: classifies the query, writes focused sub-queries, and activates only the agents that are needed. Non-clinical messages short-circuit to a lightweight chitchat node without running any retrieval.
- **Internet Search Agent**: searches clinical guidelines and differential diagnoses via DuckDuckGo and recent literature via the NCBI Entrez/PubMed API (PMID, title, abstract, authors).
- **RAG Agent**: retrieves similar radiology cases from the knowledge base and summarizes the clinically relevant ones.
- **Reasoning Agent**: merges all evidence into one structured answer in which every factual claim cites a source.
- The Internet Search and RAG agents run **in parallel** (LangGraph `Send` fan-out). **MemorySaver** checkpointing keeps multi-turn conversation state per session.

## Retrieval pipeline
 
| Component | Choice |
|---|---|
| Corpus | Radiology case texts + figure captions from the MultiCaRe dataset |
| Chunking | One document per case (keeps the clinical narrative intact) + a separate caption document |
| Embeddings | OpenAI `text-embedding-3-small` (1,536-d) |
| Vector store | ChromaDB (persistent, with metadata: imaging modality, anatomical region, view) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (k = 60, dense 0.7 / sparse 0.3) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (top-50 → top-10) |
 
Four strategies are implemented and selectable in `config.py`: `dense`, `hybrid`, `ensemble` and `multi_query`.


## Evaluation
 
The pipeline was evaluated on a clinical benchmark covering pulmonology, neuro-oncology, oncology, rheumatology, emergency medicine and abdominal radiology. Two comparisons were run:
 
- **Retrieval strategies** (dense, hybrid, ensemble, multi-query): hybrid retrieval was selected because it gave the best faithfulness and context precision, the most safety-critical metrics in a clinical setting.
- **LLM backbones** (GPT-4o-mini, Mistral-7B-Instruct, Llama3-8B-Instruct): compared under identical prompts, tools and retrieval settings.
Metrics: faithfulness, answer relevancy, context precision/recall, citation correctness/coverage/F1, BERTScore F1 and LLM-as-a-Judge. 

## Tech stack
 
| Category | Tools |
|---|---|
| **Language** | Python |
| **Agent orchestration** | LangGraph (StateGraph, parallel `Send` fan-out, MemorySaver checkpointing), LangChain |
| **LLMs** | OpenAI GPT-4o / GPT-4o-mini, Mistral (`mistral-small`, Mistral-7B-Instruct), Llama 3 8B via Ollama |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector store** | ChromaDB (persistent, metadata filtering) |
| **Retrieval** | BM25 (`rank-bm25`), Reciprocal Rank Fusion, multi-query reformulation |
| **Reranking** | Sentence-Transformers cross-encoder (`ms-marco-MiniLM-L-6-v2`), PyTorch |
| **External evidence** | NCBI Entrez / PubMed API (Biopython), DuckDuckGo Search |
| **Observability** | LangSmith tracing |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Data processing** | pandas, openpyxl, pypdf, python-docx |
