import hashlib
import fitz
from cpc_agent.models import Paper


def _short_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]


def _extract_title_from_text(text: str) -> str:
    """Use the first non-empty line of the paper as fallback title."""
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) > 10:
            return stripped[:120]
    return "Unknown Title"


def extract_paper(path: str) -> Paper:
    doc = fitz.open(path)
    try:
        meta = doc.metadata or {}
        meta_title = (meta.get("title") or "").strip()
        meta_author = (meta.get("author") or "").strip()

        pages_text = [page.get_text() for page in doc]
        full_text = "\n".join(pages_text)
    finally:
        doc.close()

    title = meta_title if meta_title else _extract_title_from_text(full_text)
    authors = meta_author if meta_author else "Unknown Authors"

    return Paper(
        id=_short_id(path),
        path=path,
        title=title,
        authors=authors,
        full_text=full_text,
    )
