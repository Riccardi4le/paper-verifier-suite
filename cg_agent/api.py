"""OpenAlex API client for fetching paper metadata and references."""

from __future__ import annotations
import os
import re
import requests
from typing import Optional

OPENALEX_BASE = "https://api.openalex.org"
_EMAIL = os.getenv("OPENALEX_EMAIL", "research@example.com")
_TIMEOUT = 12


def _headers() -> dict:
    return {"User-Agent": f"CG-Agent/1.0 (mailto:{_EMAIL})"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _score_work_match(query: str, work: dict) -> float:
    normalized_query = _norm(query)
    if not normalized_query:
        return 0.0

    title = work.get("title") or ""
    normalized_title = _norm(title)
    score = 0.0

    if normalized_title == normalized_query:
        score += 100.0
    elif normalized_query and normalized_query in normalized_title:
        score += 40.0
    elif normalized_title and normalized_title in normalized_query:
        score += 25.0

    query_tokens = set(normalized_query.split())
    title_tokens = set(normalized_title.split())
    if query_tokens and title_tokens:
        score += (len(query_tokens & title_tokens) / len(query_tokens)) * 30.0

    year = work.get("publication_year")
    if year and str(year) in query:
        score += 5.0

    authorships = work.get("authorships") or []
    author_names = [
        _norm((auth.get("author") or {}).get("display_name", ""))
        for auth in authorships[:3]
    ]
    if any(name and name in normalized_query for name in author_names):
        score += 8.0

    if work.get("doi"):
        score += 1.0

    return score


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, pos_list in inverted_index.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _url_to_query(url: str) -> str:
    """Extract a DOI, PMID, or arXiv ID from an academic URL."""
    # doi.org/10.xxx/...
    m = re.search(r'doi\.org/(10\.\d{4,}/[^\s?&#]+)', url)
    if m:
        return m.group(1).rstrip('.,;)')
    # Any DOI pattern embedded in URL path/query
    m = re.search(r'(10\.\d{4,}/[^\s?&#]+)', url)
    if m:
        return m.group(1).rstrip('.,;)')
    # PubMed: pubmed.ncbi.nlm.nih.gov/12345678
    m = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', url)
    if m:
        return f"pmid:{m.group(1)}"
    # arXiv: arxiv.org/abs/2301.12345 → use arXiv DOI (reliable on OpenAlex)
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)', url)
    if m:
        return f"10.48550/arxiv.{m.group(1)}"
    return url


def fetch_work_openalex(query: str) -> Optional[dict]:
    """Resolve a citation string, URL, or DOI to an OpenAlex work dict."""
    query = query.strip()
    if query.startswith("http://") or query.startswith("https://"):
        query = _url_to_query(query)

    # PubMed ID
    pmid_match = re.match(r'pmid:(\d+)', query)
    if pmid_match:
        params = {
            "filter": f"ids.pmid:{pmid_match.group(1)}",
            "per-page": 1,
            "select": (
                "id,title,authorships,publication_year,doi,"
                "abstract_inverted_index,referenced_works,type"
            ),
        }
        try:
            resp = requests.get(
                f"{OPENALEX_BASE}/works", params=params, headers=_headers(), timeout=_TIMEOUT
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    return results[0]
        except requests.RequestException:
            pass

    # arXiv ID
    arxiv_match = re.match(r'arxiv:([0-9]{4}\.[0-9]+)', query)
    if arxiv_match:
        params = {
            "filter": f"ids.arxiv:{arxiv_match.group(1)}",
            "per-page": 1,
            "select": (
                "id,title,authorships,publication_year,doi,"
                "abstract_inverted_index,referenced_works,type"
            ),
        }
        try:
            resp = requests.get(
                f"{OPENALEX_BASE}/works", params=params, headers=_headers(), timeout=_TIMEOUT
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    return results[0]
        except requests.RequestException:
            pass

    # Try DOI first
    doi_match = re.search(r'10\.\d{4,}/\S+', query)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;)")
        url = f"{OPENALEX_BASE}/works/https://doi.org/{doi}"
        try:
            resp = requests.get(url, headers=_headers(), timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass

    # Try OpenAlex ID directly
    oaid_match = re.search(r'W\d{6,}', query)
    if oaid_match:
        url = f"{OPENALEX_BASE}/works/{oaid_match.group(0)}"
        try:
            resp = requests.get(url, headers=_headers(), timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass

    # Fall back to text search
    params = {
        "search": query,
        "per-page": 3,
        "select": (
            "id,title,authorships,publication_year,doi,"
            "abstract_inverted_index,referenced_works,type"
        ),
    }
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/works", params=params, headers=_headers(), timeout=_TIMEOUT
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None
        ranked = sorted(results, key=lambda work: _score_work_match(query, work), reverse=True)
        return ranked[0]
    except requests.RequestException:
        return None


def fetch_referenced_works_metadata(work_ids: list[str]) -> list[dict]:
    """Batch-fetch metadata for up to 50 referenced work IDs (preserves order)."""
    if not work_ids:
        return []
    ids = work_ids[:50]
    # Strip URL prefix → bare IDs
    bare = [wid.replace("https://openalex.org/", "") for wid in ids]
    filter_str = "|".join(bare)
    params = {
        "filter": f"openalex_id:{filter_str}",
        "per-page": 50,
        "select": "id,title,authorships,publication_year,doi",
    }
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/works", params=params, headers=_headers(), timeout=15
        )
        if resp.status_code != 200:
            return []
        fetched = {w["id"]: w for w in resp.json().get("results", [])}
        # Return in original order, fill missing with stub
        ordered = []
        for wid in ids:
            full_id = wid if wid.startswith("https://") else f"https://openalex.org/{wid}"
            ordered.append(fetched.get(full_id, {"id": full_id, "title": "Unknown"}))
        return ordered
    except requests.RequestException:
        return []


def work_to_paper(work: dict) -> dict:
    """Convert raw OpenAlex work dict to PaperData dict."""
    authors = [
        auth.get("author", {}).get("display_name", "")
        for auth in (work.get("authorships") or [])
        if auth.get("author", {}).get("display_name")
    ]
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    return {
        "paper_id": work.get("id", ""),
        "openalex_id": work.get("id", ""),
        "title": work.get("title") or "Unknown",
        "authors": authors,
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "abstract": abstract,
        "source": "openalex",
        "referenced_work_ids": work.get("referenced_works", []),
    }


def format_refs_for_llm(referenced_works: list[dict]) -> str:
    """Format referenced works as a numbered list for LLM prompts."""
    lines = []
    for i, rw in enumerate(referenced_works, start=1):
        authors = rw.get("authorships") or []
        author_names = [
            name for name in (
                (a.get("author") or {}).get("display_name", "") for a in authors[:2]
            )
            if name
        ]
        if author_names:
            tokens = author_names[0].split()
            first = tokens[-1] if tokens else author_names[0]
        else:
            first = "Unknown"
        suffix = " et al." if len(author_names) > 1 else ""
        year = rw.get("publication_year", "?")
        title = rw.get("title") or "Unknown"
        lines.append(f"[{i}] {title} — {first}{suffix} ({year})")
    return "\n".join(lines) if lines else "References not available"
