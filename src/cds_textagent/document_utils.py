import io

MAX_DOC_CHARS = 12_000


def extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(uploaded_file) -> tuple[str, str | None]:
    """Return (extracted_text, error_message). error_message is None on success."""
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            text = extract_pdf(raw)
        elif name.endswith(".docx"):
            text = extract_docx(raw)
        elif name.endswith(".doc"):
            return "", "Legacy .doc format is not supported — please save as .docx and re-upload."
        else:
            return "", f"Unsupported file type: {uploaded_file.name}"
    except Exception as exc:
        return "", f"Could not read file: {exc}"

    text = text.strip()
    if not text:
        return "", "No readable text found in the file."
    truncated = len(text) > MAX_DOC_CHARS
    return text[:MAX_DOC_CHARS], "truncated" if truncated else None
