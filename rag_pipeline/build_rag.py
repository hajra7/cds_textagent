"""
Unified RAG: Excel → Entities → Chroma (cosine) + BM25 → HybridRetriever/RAG Chain
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import logging
import re  # ← FIXED: was 'regex as re'
from typing import List, Dict, Optional
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # ← ADDED
from rank_bm25 import BM25Okapi

from config import EXCEL_FILE_PATH, PROCESSED_DATA_DIR, ENTITY_EXTRACTION_MODEL
from rag_pipeline.text_utils import clean_text  # Your utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Fixed: no .apply on str."""
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'\s+', ' ', text)  # ← Now works!
    text = re.sub(r'\[Figure \d+\w*\]', '[FIGURE]', text)
    return text.strip()

def load_clean_df(excel_path: Path) -> pd.DataFrame:
    logger.info(f"Loading {excel_path}")
    df = pd.read_excel(excel_path)
    logger.info(f"Shape: {df.shape}")
    cols = ['patient_id', 'case_id', 'case_text', 'article_id', 'caption', 'radiology_region_granular']
    df = df[cols].copy()
    df['case_text'] = df['case_text'].astype(str).apply(clean_text)
    df['caption'] = df['caption'].fillna('').astype(str).apply(clean_text)
    df = df[df['case_text'].str.len() >= 20]
    df.to_parquet(PROCESSED_DATA_DIR / "cases.parquet")
    logger.info(f"Cleaned: {len(df)} rows → cases.parquet")
    return df

# ← ADDED: Pydantic schema for structured output
class Entities(BaseModel):
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="M/F/None")
    radiology_region: Optional[str] = Field(None, description="thorax/abdomen/etc")

def extract_entities(text: str) -> Dict:
    """LLM entities with structured output."""
    llm = ChatOpenAI(model=ENTITY_EXTRACTION_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    From clinical text, extract: age (int/None), gender ("M"/"F"/None), radiology_region (str/None).
    JSON only.
    Text: {text}
    """)
    chain = prompt | llm.with_structured_output(Entities)  # ← FIXED: BaseModel schema
    try:
        result = chain.invoke({"text": text[:4000]})
        return result.model_dump()  # ← Convert Pydantic model to dict
    except Exception as e:
        logger.warning(f"Entity extract fail: {e}")
        return {"age": None, "gender": None, "radiology_region": None}

def df_to_documents(df: pd.DataFrame) -> tuple[List[Document], List[str]]:
    logger.info("Building Documents w/ entities...")
    docs, texts = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    llm_entities = {}  # Cache per case_id

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row['case_id']
        if case_id not in llm_entities:
            llm_entities[case_id] = extract_entities(row['case_text'])

        content = f"{row['case_text']} | Caption: {row['caption']}"
        splits = splitter.split_text(content) if len(content) > 1500 else [content]

        for i, split in enumerate(splits):
            metadata = {
                'patient_id': row['patient_id'],
                'case_id': case_id,
                'article_id': row['article_id'],
                'radiology_region_granular': row['radiology_region_granular'],
                'chunk_id': f"{case_id}_{i}",
                **llm_entities[case_id]  # age/gender/region for filter
            }
            docs.append(Document(page_content=split, metadata=metadata))
            texts.append(split)  # For BM25

    logger.info(f"✓ {len(docs)} Documents (entities cached)")
    return docs, texts

def build_hybrid_index(docs: List[Document], texts: List[str], index_path: Path):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # === CHROMA COSINE (YOUR REQ) ===
    chroma_path = index_path / "chroma_db"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(chroma_path),
        collection_metadata={"hnsw:space": "cosine"}
    )
    vectorstore.persist()
    logger.info(f"✓ Chroma cosine HNSW: {chroma_path}")

    # BM25 sparse
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(index_path / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts}, f)
    logger.info(f"BM25 saved: {index_path / 'bm25.pkl'}")

    # Metadata Parquet
    pd.DataFrame([d.metadata for d in docs]).to_parquet(index_path / "metadata.parquet")

def main():
    use_llm = '--no-llm' not in sys.argv
    logger.info(f"RAG Build (LLM entities: {use_llm})")
    index_path = PROCESSED_DATA_DIR / "hybrid_index"
    index_path.mkdir(exist_ok=True)

    df = load_clean_df(EXCEL_FILE_PATH)
    docs, texts = df_to_documents(df)
    build_hybrid_index(docs, texts, index_path)

    # Stats
    lengths = [len(d.page_content) for d in docs]
    logger.info(f"Chunks: {len(docs)} | Avg: {np.mean(lengths):.0f} | Regions: {df['radiology_region_granular'].value_counts().head()}")

if __name__ == "__main__":
    main()
