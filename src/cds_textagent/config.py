"""
Configuration file for CDSS Agent System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data
EXCEL_FILE_PATH = RAW_DATA_DIR / "radiology_cases.xlsx"

# Processed data
CASES_JSONL_PATH = PROCESSED_DATA_DIR / "cases.jsonl"
CHUNKS_JSONL_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
META_CHUNKS_JSONL_PATH = PROCESSED_DATA_DIR / "meta_chunks.jsonl"
# Chroma paths (remove old FAISS_INDEX_PATH etc.)
CHROMA_DB_PATH = PROCESSED_DATA_DIR / "hybrid_index" / "chroma_db"
BM25_PATH = PROCESSED_DATA_DIR / "hybrid_index" / "bm25.pkl"


# ============================================================================
# DATASET COLUMNS (from Excel)
# ============================================================================
DATASET_COLUMNS = {
    'patient_id': 'patient_id',
    'case_text': 'case_text',
    'article_id': 'article_id',
    'caption': 'caption',
    'radiology_region_granular': 'radiology_region_granular',
}

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
# OpenAI Embeddings (text-embedding-3-small)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension

# ============================================================================
# CHUNKING PARAMETERS
# ============================================================================
# Strategy: One chunk per case_text (no splitting)
CHUNK_STRATEGY = "one_per_case"  # Don't split case_text
CHUNK_SIZE = 0  # Not used for one_per_case strategy
CHUNK_OVERLAP = 0  # Not used for one_per_case strategy

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================
# Initial metadata filtering
TOP_K_INITIAL = 50  # Filter top 50 using metadata

# Final retrieval after cosine similarity
TOP_K_FINAL = 10  # Return top 10 after reranking

# Similarity threshold (0-1)
SIMILARITY_THRESHOLD = 0.5

# Hybrid retrieval weights
DENSE_WEIGHT = 0.7  # Weight for dense (embedding) retrieval
SPARSE_WEIGHT = 0.3  # Weight for sparse (BM25) retrieval

# Retrieval strategy: "dense" | "hybrid" | "ensemble" | "multi_query"
RETRIEVAL_STRATEGY = "dense"

# Multi-query retrieval
MULTI_QUERY_NUM_REFORMULATIONS = 3

# Reciprocal Rank Fusion constant (for hybrid/ensemble score fusion)
RRF_K = 60

# ============================================================================
# ENTITY EXTRACTION (for metadata filtering)
# ============================================================================
ENTITY_EXTRACTION_MODEL = "mistral-small-latest"
# ENTITY_EXTRACTION_MODEL = "gpt-4o-mini"
ENTITY_EXTRACTION_TEMPERATURE = 0.0

# Entities to extract from case_text
ENTITIES_TO_EXTRACT = ['age', 'gender', 'radiology_region']

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_MODEL = "mistral-small-latest"
# LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1500

# Pluggable LLM backend — default: Mistral native API (api.mistral.ai)
# Set MISTRAL_API_KEY in your .env to authenticate.
# Alternative models (override via .env):
#   Llama (Ollama) : AGENT_LLM_MODEL=llama3:8b-instruct  AGENT_LLM_BASE_URL=http://localhost:11434/v1  AGENT_LLM_API_KEY=ollama
#   GPT (OpenAI)  : AGENT_LLM_MODEL=gpt-4o              AGENT_LLM_BASE_URL=                           AGENT_LLM_API_KEY=<openai-key>
AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "mistral-small-latest")
AGENT_LLM_BASE_URL = os.getenv("AGENT_LLM_BASE_URL") or None  # Not used for native Mistral
AGENT_LLM_API_KEY = os.getenv("AGENT_LLM_API_KEY") or os.getenv("MISTRAL_API_KEY")

# Planner-specific LLM (can differ from agent LLM)
# Alternatives: PLANNER_LLM_MODEL=gpt-4o-mini  (OpenAI)
#               PLANNER_LLM_MODEL=llama3:8b-instruct  PLANNER_LLM_BASE_URL=http://localhost:11434/v1  (Llama via Ollama)
PLANNER_LLM_MODEL = os.getenv("PLANNER_LLM_MODEL", "mistral-small-latest")
PLANNER_LLM_BASE_URL = os.getenv("PLANNER_LLM_BASE_URL") or None  # Not used for native Mistral
PLANNER_LLM_API_KEY = os.getenv("PLANNER_LLM_API_KEY") or os.getenv("MISTRAL_API_KEY")

# ============================================================================
# API KEYS (from environment variables)
# ============================================================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "cdss-agent")

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
APP_TITLE = "Clinical Decision Support System"
APP_ICON = "🏥"

# ============================================================================
# SAFETY & CONFIDENCE
# ============================================================================
MIN_CONFIDENCE_THRESHOLD = 0.6
UNCERTAINTY_MESSAGE = "I don't have enough evidence to answer this confidently."

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "app.log"
