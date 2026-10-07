"""Groq-powered analysis tools for citation verification."""

from __future__ import annotations
import json
import os
from groq import Groq

_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# gpt-oss models reason before answering, and reasoning tokens count against
# max_tokens — reserve headroom so the JSON answer is never truncated.
_REASONING_HEADROOM = 2048

_SYSTEM = """You are an expert scientific citation analyst. Your tasks:
1. Locate specific claims in academic paper abstracts
2. Identify which reference is cited to support the claim
3. Judge whether the citing paper accurately represents the cited source
4. Determine if a paper presents original empirical data (primary source)

Be precise. Quote specific text from the paper. Respond with valid JSON only."""


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set — add it to your .env file")
    # Free tier has tight per-minute token limits: let the SDK back off on 429s.
    return Groq(api_key=api_key, max_retries=6)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def analyze_paper(
    original_claim: str,
    reference_claim: str,
    paper: dict,
    refs_text: str,
    client: Groq | None = None,
) -> dict:
    if client is None:
        client = _client()

    title = paper.get("title", "Unknown")
    year = paper.get("year", "?")
    abstract = (paper.get("abstract") or "")[:2500]
    excerpt = abstract if abstract else "ABSTRACT NOT AVAILABLE"

    prompt = f"""Analyze this academic paper to verify a citation chain.

We only have the paper title, abstract, and reference list metadata. Do not claim to have inspected the full paper.

CLAIM BEING CHASED
Original claim (from user's paper): "{original_claim}"
Version cited in the parent paper: "{reference_claim}"

PAPER METADATA / ABSTRACT EXCERPT
Title: {title} ({year})
Abstract: {excerpt}

REFERENCES LISTED IN THIS PAPER
{refs_text}

YOUR TASKS
1. Based only on the title and abstract excerpt, does this paper contain the claim or a closely related version?
   - If YES: quote the exact text
   - If NO: explain briefly and stay conservative
2. If the claim is found in the excerpt, which reference in the list above does this paper most likely cite to support it?
   Give the reference TITLE (not number) so it can be searched independently.
3. Compare "Version cited in the parent paper" with what THIS paper says in the available excerpt.
   Choose one verdict:
     supported    -> accurately represented
     partial      -> key nuance dropped or simplified
     distorted    -> significantly changed, exaggerated, or minimized
     contradicted -> the paper says the opposite
     not_found    -> claim not present in the available excerpt
4. Is this paper a PRIMARY SOURCE for the claim, based on the available excerpt?
   Primary = reports original empirical data (n=X, our experiment, our study, we found...)
   NOT primary = review, meta-analysis, cites someone else for this finding

Respond ONLY with this JSON (no markdown fences, no extra text):
{{
  "claim_found": true or false,
  "claim_text": "exact quote from the available excerpt, or null",
  "verdict": "supported|partial|distorted|contradicted|not_found",
  "reasoning": "one concise sentence",
  "is_primary": true or false,
  "evidence_type": "empirical|review|meta-analysis|theoretical|unclear",
  "next_ref_title": "exact title from the reference list above, or null if primary/not_found",
  "next_ref_year": year as integer or null
}}"""

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=768 + _REASONING_HEADROOM,
        extra_body={"reasoning_effort": "medium"},
    )

    result = _extract_json(response.choices[0].message.content or "")
    if not result:
        return {
            "claim_found": False,
            "claim_text": None,
            "verdict": "not_found",
            "reasoning": "LLM response could not be parsed",
            "is_primary": False,
            "evidence_type": "unclear",
            "next_ref_title": None,
            "next_ref_year": None,
        }
    return result
