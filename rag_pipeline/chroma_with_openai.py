"""
ChromaDB (OpenAI) builder — 2 docs per row, streaming batches from Excel

Fixes:
- Proper whitespace regex: r"\s+"
- Global row_id + doc_id uniqueness across batches
- Better rate-limit detection (OpenAIError status_code==429 + fallback)
"""

import logging
import random
import re
import shutil
import time
from pathlib import Path
from typing import Iterator, List, Tuple
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ


import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import EXCEL_FILE_PATH, PROCESSED_DATA_DIR

# Rate-limit detection helper (works for openai>=1.x)
try:
    from openai import OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    OpenAIError = Exception  # fallback so code still runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------------------------
# Config knobs
# -------------------------
EXCEL_BATCH_SIZE = 500
MAX_RETRIES = 8
BASE_DELAY_SEC = 1.5
MAX_DELAY_SEC = 30

MIN_CASE_TEXT_LEN = 20
MIN_CAPTION_LEN = 10


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)  # <-- correct regex
    text = re.sub(r"\[Figure \d+\w*\]", "[FIGURE]", text)
    return text.strip()


def to_json_safe(x):
    """Chroma metadata must be JSON-serializable: str/int/float/bool/None."""
    if pd.isna(x):
        return None
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def is_rate_limit_error(e: Exception) -> bool:
    # Structured OpenAI errors (best-case)
    try:
        if isinstance(e, OpenAIError) and getattr(e, "status_code", None) == 429:
            return True
    except Exception:
        pass

    # Fallback: string heuristics
    msg = str(e).lower()
    return any(kw in msg for kw in ["rate limit", "429", "too many requests"])


def add_with_backoff(vectorstore: Chroma, batch_docs: List[Document]):
    attempt = 0
    while True:
        try:
            vectorstore.add_documents(batch_docs)
            return
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES or not is_rate_limit_error(e):
                logger.exception(f"Batch failed after {attempt} attempts.")
                raise

            delay = min(MAX_DELAY_SEC, BASE_DELAY_SEC * (2 ** (attempt - 1)))
            delay = delay * (0.7 + 0.6 * random.random())  # jitter in [0.7, 1.3]
            logger.warning(f"Rate limited. Retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay)


def doc_generator(excel_path: Path, batch_size: int = 500) -> Iterator[Tuple[List[Document], int]]:
    """
    Yield docs in batches.
    Each row yields up to 2 docs: case_text and caption.

    IMPORTANT:
    - We reset index to make a stable global_row_id (0..N-1).
    - doc_id uses global_row_id to avoid duplicates across batches.
    """
    df = pd.read_excel(excel_path)

    needed = [
        "file_id",
        "case_id",
        "case_text",
        "article_id",
        "file",
        "image_component",
        "caption",
        "image_type",
        "image_subtype",
        "radiology_region",
        "radiology_region_granular",
        "radiology_view",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in Excel: {missing}\nAvailable columns: {list(df.columns)}")

    df = df[needed].copy().reset_index(drop=True)  # <-- stable global indexing

    df["case_text"] = df["case_text"].apply(clean_text)
    df["caption"] = df["caption"].fillna("").apply(clean_text)

    total_rows = len(df)
    for start in range(0, total_rows, batch_size):
        batch_df = df.iloc[start : start + batch_size]
        batch_docs: List[Document] = []

        # enumerate ensures 0..len(batch_df)-1; add start => global_row_id
        for j, (_, row) in enumerate(batch_df.iterrows()):
            global_row_id = start + j

            base_meta = {
                "case_id": to_json_safe(row["case_id"]),
                "file_id": to_json_safe(row["file_id"]),
                "article_id": to_json_safe(row["article_id"]),
                "file": to_json_safe(row["file"]),
                "image_component": to_json_safe(row["image_component"]),
                "image_type": to_json_safe(row["image_type"]),
                "image_subtype": to_json_safe(row["image_subtype"]),
                "radiology_region": to_json_safe(row["radiology_region"]),
                "radiology_region_granular": to_json_safe(row["radiology_region_granular"]),
                "radiology_view": to_json_safe(row["radiology_view"]),
                "row_id": global_row_id,
            }

            case_text = row["case_text"]
            if len(case_text) >= MIN_CASE_TEXT_LEN:
                meta = dict(base_meta)
                meta["doc_type"] = "case_text"
                meta["doc_id"] = f"{base_meta['case_id']}::case_text::{global_row_id}"
                batch_docs.append(Document(page_content=case_text, metadata=meta))

            caption = row["caption"]
            if len(caption) >= MIN_CAPTION_LEN:
                meta = dict(base_meta)
                meta["doc_type"] = "caption"
                meta["doc_id"] = f"{base_meta['case_id']}::caption::{global_row_id}"
                batch_docs.append(Document(page_content=caption, metadata=meta))

        batch_num = start // batch_size + 1
        yield batch_docs, batch_num


def main():
    persist_dir = PROCESSED_DATA_DIR / "chroma_openai_text_embedding_3_small"

    # Fresh build
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        logger.info("Wiped existing Chroma directory for a fresh build.")
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="cases_openai",
        collection_metadata={"hnsw:space": "cosine"},
    )

    total_docs = 0
    batch_count = 0

    for batch_docs, bnum in doc_generator(EXCEL_FILE_PATH, batch_size=EXCEL_BATCH_SIZE):
        if not batch_docs:
            logger.info(f"Batch {bnum}: 0 docs (skipped)")
            continue

        logger.info(f"Adding batch {bnum}: {len(batch_docs)} docs (running total: {total_docs})")
        add_with_backoff(vectorstore, batch_docs)

        total_docs += len(batch_docs)
        batch_count += 1

        

    logger.info(f"✅ Added {total_docs} docs in {batch_count} batches")
    logger.info(f"✅ ChromaDB path: {persist_dir}")


if __name__ == "__main__":
    main()
