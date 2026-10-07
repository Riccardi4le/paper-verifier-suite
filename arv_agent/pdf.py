"""PDF and URL ingestion for the ARV agent."""

from __future__ import annotations
import io
import re
from pathlib import Path
from typing import Optional


def ingest_paper(source: str) -> tuple[str, dict]:
    """
    Route to the appropriate extractor based on source string.
    Returns (full_text, metadata_dict).
    """
    source = source.strip()

    # ArXiv abstract URL → redirect to PDF
    m = re.match(r'https?://arxiv\.org/abs/([\d.v]+)', source)
    if m:
        source = f"https://arxiv.org/pdf/{m.group(1)}.pdf"

    if source.startswith("http://") or source.startswith("https://"):
        if source.lower().endswith(".pdf"):
            return _fetch_pdf_url(source)
        return _extract_from_url(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if path.suffix.lower() == ".pdf":
        return _extract_from_pdf(str(path))

    # Plain text fallback
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, _metadata_from_text(text, source=source, title=path.stem)


# ── PDF ──────────────────────────────────────────────────────────────────────

def _extract_from_pdf(path: str) -> tuple[str, dict]:
    with open(path, "rb") as f:
        data = f.read()
    return _pdf_bytes_to_text(data, source=path)


def _fetch_pdf_url(url: str) -> tuple[str, dict]:
    import requests
    headers = {"User-Agent": "ARV-Agent/1.0 (academic research tool)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return _pdf_bytes_to_text(resp.content, source=url)


def _pdf_bytes_to_text(data: bytes, source: str) -> tuple[str, dict]:
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf required: pip install pypdf")

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)

    info = reader.metadata or {}
    metadata = {
        "title": (info.get("/Title") or "").strip(),
        "authors": _parse_authors(info.get("/Author") or ""),
        "year": _extract_year(full_text),
        "doi": _extract_doi(full_text),
        "abstract": _extract_abstract(full_text),
        "source": source,
    }
    return full_text, metadata


# ── HTML / URL ────────────────────────────────────────────────────────────────

def _extract_from_url(url: str) -> tuple[str, dict]:
    import requests
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 required: pip install beautifulsoup4")

    headers = {"User-Agent": "ARV-Agent/1.0 (academic research tool)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)

    page_title = soup.title.string.strip() if soup.title else ""
    metadata = {
        "title": page_title,
        "authors": [],
        "year": _extract_year(text),
        "doi": _extract_doi(text),
        "abstract": _extract_abstract(text),
        "source": url,
    }
    return text, metadata


# ── helpers ───────────────────────────────────────────────────────────────────

def _metadata_from_text(text: str, source: str = "", title: str = "") -> dict:
    return {
        "title": title,
        "authors": [],
        "year": _extract_year(text),
        "doi": _extract_doi(text),
        "abstract": _extract_abstract(text),
        "source": source,
    }


def _parse_authors(author_str: str) -> list[str]:
    if not author_str:
        return []
    return [a.strip() for a in re.split(r'[;,]', author_str) if a.strip()]


def _extract_year(text: str) -> Optional[int]:
    m = re.search(r'\b(19|20)\d{2}\b', text)
    return int(m.group(0)) if m else None


def _extract_doi(text: str) -> Optional[str]:
    m = re.search(r'10\.\d{4,}/[^\s"<>\]]+', text)
    if m:
        return m.group(0).rstrip(".,;)")
    return None


def _extract_abstract(text: str) -> str:
    m = re.search(
        r'(?i)\babstract\b\s*\n+(.*?)(?=\n+(?:introduction|keywords?|1[\.\s]|background))',
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()[:2000]
    return text[:800].strip()
