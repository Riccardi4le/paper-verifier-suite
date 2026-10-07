"""Groq-powered adversarial checks for the ARV agent."""

from __future__ import annotations
import json
import os
from groq import Groq

FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
DEEP_MODEL = os.getenv("GROQ_DEEP_MODEL", "openai/gpt-oss-120b")

# gpt-oss models reason before answering, and reasoning tokens count against
# max_tokens — reserve headroom so the JSON answer is never truncated.
_REASONING_HEADROOM = 2048

_SYSTEM = """You are an adversarial academic reviewer — a harsh but fair critic in the style of a Nature/Science desk reject. Your job is NOT to summarize or praise. Your job is to find problems: methodological flaws, unsupported claims, statistical errors, and quality signals that suggest low-quality or fabricated research.

Be specific. Quote exact text when possible. Do not invent findings that are not supported by the paper excerpt provided.

Respond with valid JSON only. No markdown fences, no preamble, no explanation outside the JSON."""


def _client() -> Groq:
    # Free tier has tight per-minute token limits: let the SDK back off on 429s.
    return Groq(max_retries=6)


def _extract_json(text: str) -> dict | list:
    text = text.strip()
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def _call_llm(
    client: Groq,
    *,
    model: str,
    max_tokens: int,
    prompt: str,
    reasoning_effort: str = "medium",
) -> str | None:
    """Single point of contact with the Groq API."""
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens + _REASONING_HEADROOM,
            extra_body={"reasoning_effort": reasoning_effort},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty response")
        return text
    except Exception as exc:
        raise RuntimeError(f"Groq call failed for model {model}: {exc}") from exc


# ── Quick classification / extraction (fast model) ───────────────────────────

def classify_paper(
    text: str,
    metadata: dict,
    client: Groq | None = None,
) -> dict:
    if client is None:
        client = _client()

    abstract = metadata.get("abstract") or text[:1200]
    prompt = f"""Classify this academic paper.

Title: {metadata.get("title", "Unknown")}
Abstract/beginning:
{abstract[:1500]}

Respond ONLY with this JSON:
{{
  "paper_type": "empirical|review|meta_analysis|theoretical|unclear",
  "reasoning": "one sentence"
}}"""

    try:
        text = _call_llm(client, model=FAST_MODEL, max_tokens=256, prompt=prompt, reasoning_effort="low")
    except RuntimeError:
        return {"paper_type": "unclear", "reasoning": "Classification API call failed"}
    result = _extract_json(text)
    if not isinstance(result, dict) or "paper_type" not in result:
        return {"paper_type": "unclear", "reasoning": "Classification failed"}
    return result


def extract_claims(
    text: str,
    metadata: dict,
    client: Groq | None = None,
) -> list:
    if client is None:
        client = _client()

    abstract = metadata.get("abstract") or ""
    sample = (abstract + "\n" + text[:3000])[:4000]

    prompt = f"""Extract the 5 most important verifiable claims from this paper.

Paper excerpt:
{sample}

For each claim, identify:
- claim_text: exact quote or close paraphrase
- location: abstract / intro / methods / results / conclusion
- type: factual / statistical / causal / comparative

Respond ONLY with a JSON array:
[
  {{"claim_text": "...", "location": "abstract", "type": "statistical"}},
  ...
]"""

    try:
        text = _call_llm(client, model=FAST_MODEL, max_tokens=1024, prompt=prompt, reasoning_effort="low")
    except RuntimeError:
        return []
    result = _extract_json(text)
    return result if isinstance(result, list) else []


# ── Dispatcher ────────────────────────────────────────────────────────────────

def run_check(
    check_type: str,
    paper_text: str,
    metadata: dict,
    claims: list,
    grim_info: dict | None = None,
    pvalue_info: dict | None = None,
    client: Groq | None = None,
) -> list:
    if client is None:
        client = _client()

    dispatch = {
        "statistical":   _check_statistical,
        "claim_support": _check_claim_support,
        "citation":      _check_citation,
        "language":      _check_language,
        "methodology":   _check_methodology,
    }
    fn = dispatch.get(check_type)
    if fn is None:
        return []
    return fn(paper_text, metadata, claims, grim_info, pvalue_info, client)


# ── Finding format (shared instruction injected into every check prompt) ──────

_FINDING_SCHEMA = """
Return a JSON array of findings. Each element MUST have these exact keys:
{
  "check_type": "<same as this check>",
  "severity": "low|medium|high|critical",
  "title": "8 words max",
  "evidence": "exact quote or specific observation",
  "explanation": "why this is a problem, 1-2 sentences"
}
Return [] if no issues are found. Respond ONLY with the JSON array."""


# ── Individual check functions ────────────────────────────────────────────────

def _check_statistical(text, metadata, claims, grim_info, pvalue_info, client):
    grim_ctx = ""
    if grim_info and grim_info.get("inconsistent", 0) > 0:
        grim_ctx = f"\n[GRIM pre-scan] {grim_info['inconsistent']} inconsistent mean(s): {grim_info['detail']}"

    pval_ctx = ""
    if pvalue_info:
        pval_ctx = f"\n[P-value pre-scan] {pvalue_info.get('detail', '')}"
        if pvalue_info.get("suspicious"):
            pval_ctx += " ← SUSPICIOUS CLUSTERING DETECTED"

    prompt = f"""Perform a STATISTICAL adversarial review of this paper.
{grim_ctx}{pval_ctx}

Paper (first 8000 chars):
{text[:8000]}

Look for:
1. P-hacking: many p-values just below 0.05 (flag if pre-scan already detected this)
2. GRIM failures: reported means that are arithmetically impossible for the given n
3. Sample size inconsistencies between sections (Methods n=X, Table n=Y)
4. Implausible effect sizes (Cohen's d > 2, r > 0.9 without explanation)
5. Missing measures of dispersion (SD, SE, CI) where expected
6. Test assumptions violated (e.g., t-test on obviously non-normal data, acknowledged in text)
check_type for all findings must be "statistical".
{_FINDING_SCHEMA}"""

    out = _call_llm(client, model=DEEP_MODEL, max_tokens=1500, prompt=prompt)
    if out is None:
        return []
    result = _extract_json(out)
    return result if isinstance(result, list) else []


def _check_claim_support(text, metadata, claims, grim_info, pvalue_info, client):
    claims_block = "\n".join(
        f"{i+1}. [{c.get('type','?')}] {c.get('claim_text','')}"
        for i, c in enumerate(claims[:5])
    ) or "No claims pre-extracted — infer from abstract."

    prompt = f"""Perform a CLAIM SUPPORT adversarial review of this paper.

CLAIMS TO VERIFY:
{claims_block}

Paper (first 10000 chars):
{text[:10000]}

For each claim, check:
1. Does the paper present actual data (tables, figures, numbers) supporting it?
2. Is the claim overstated relative to the evidence (e.g., "proves" vs. "suggests")?
3. Are appropriate baselines / control groups present?
4. Are obvious alternative explanations acknowledged and tested?
5. Does the abstract make claims not backed by the results section?
check_type for all findings must be "claim_support".
{_FINDING_SCHEMA}"""

    out = _call_llm(client, model=DEEP_MODEL, max_tokens=2000, prompt=prompt)
    if out is None:
        return []
    result = _extract_json(out)
    return result if isinstance(result, list) else []


def _check_citation(text, metadata, claims, grim_info, pvalue_info, client):
    ref_start = max(
        text.lower().rfind("references"),
        text.lower().rfind("bibliography"),
    )
    ref_section = text[ref_start: ref_start + 4000] if ref_start > 0 else text[-3000:]

    prompt = f"""Perform a CITATION QUALITY adversarial review of this paper.

Paper body (first 6000 chars):
{text[:6000]}

References section:
{ref_section}

Look for:
1. Self-citation clusters: disproportionate share of refs by same author(s)
2. Citation stacking: 5+ citations for a single basic claim
3. Missing counter-evidence: contested claims supported only by agreeing papers
4. Citing secondary sources when well-known primary sources exist
5. Outdated citations in fast-moving fields (e.g., 2015 papers as state-of-the-art for 2024 AI claims)
6. Circular citation networks: papers that only cite each other with no external primary source
check_type for all findings must be "citation".
{_FINDING_SCHEMA}"""

    out = _call_llm(client, model=DEEP_MODEL, max_tokens=1500, prompt=prompt)
    if out is None:
        return []
    result = _extract_json(out)
    return result if isinstance(result, list) else []


# Known tortured phrases: (bad_form, correct_term)
_TORTURED = [
    ("profound learning", "deep learning"),
    ("irregular timberland", "random forest"),
    ("counterfeit neural", "artificial neural"),
    ("hereditary calculation", "genetic algorithm"),
    ("man-made reasoning", "artificial intelligence"),
    ("fluffy rationale", "fuzzy logic"),
    ("help vector machine", "support vector machine"),
    ("brain organization", "neural network"),
    ("bunch learning", "ensemble learning"),
    ("angle inclination", "gradient descent"),
    ("convolutional neural organization", "convolutional neural network"),
    ("characteristic extraction", "feature extraction"),
]


def _check_language(text, metadata, claims, grim_info, pvalue_info, client):
    findings: list[dict] = []

    text_lower = text.lower()
    for bad, good in _TORTURED:
        if bad in text_lower:
            findings.append({
                "check_type": "language",
                "severity": "critical",
                "title": "Tortured phrase detected",
                "evidence": f'"{bad}" (standard term: "{good}")',
                "explanation": (
                    "This is a known AI-paraphrased scientific term used to evade "
                    "plagiarism detectors, a hallmark of paper mill output."
                ),
            })

    already_found = [bad for bad, _ in _TORTURED if bad in text_lower]

    prompt = f"""Perform a LANGUAGE QUALITY adversarial review of this paper.

Already detected tortured phrases: {already_found}

Paper (first 8000 chars):
{text[:8000]}

Look for:
1. Excessive hedging ("may possibly suggest that perhaps...") — only flag if pervasive, not occasional
2. Generic boilerplate conclusions that could appear in any paper on the topic — do NOT flag standard structural phrases like "The remainder of the paper is organized as follows" or "In this paper we propose"; these are universal conventions, not quality signals
3. Inconsistent terminology (same concept given different names without definition)
4. Overclaiming absolute certainty ("definitively proves", "conclusively demonstrates") where evidence is weak
5. Unnaturally dense synonym substitution suggesting heavy AI paraphrasing
6. Abstract promises results the paper does not actually deliver
Do NOT re-flag the tortured phrases already listed above — the pre-scan handled them.
check_type for all findings must be "language".
{_FINDING_SCHEMA}"""

    out = _call_llm(client, model=DEEP_MODEL, max_tokens=1500, prompt=prompt)
    if out is None:
        return findings
    result = _extract_json(out)
    if isinstance(result, list):
        already_found_lower = {b.lower() for b in already_found}
        for f in result:
            evidence = (f.get("evidence") or "").lower()
            if any(bad in evidence for bad in already_found_lower):
                continue
            findings.append(f)
    return findings


def _check_methodology(text, metadata, claims, grim_info, pvalue_info, client):
    prompt = f"""Perform a METHODOLOGY & TRANSPARENCY adversarial review of this paper.

Paper (first 10000 chars):
{text[:10000]}

IMPORTANT: Apply standards appropriate to the paper's field.
- Pre-registration is NOT a standard requirement in computer science, systems, or machine learning research. Do NOT flag its absence unless the paper explicitly frames itself as a confirmatory clinical or behavioural study.
- Ethics approval is NOT required for CS/ML/systems papers unless the work directly involves human subjects or animals. Do NOT flag its absence for papers about algorithms, benchmarks, or software systems.

Look for:
1. Pre-registration: flag ONLY for papers that are explicitly confirmatory studies in medicine, psychology, or social science — not for CS/ML/systems papers
2. Data availability: no data sharing statement or repository link (for papers with novel datasets)
3. Code availability: no code/script sharing for computational work with novel implementations
4. Conflict of interest: missing COI statement, or COI that should raise concerns
5. Ethics approval: flag ONLY if the work involves human subjects or animals
6. Selective reporting: vague references to "additional analyses" or "supplementary results" not provided
7. Underpowered design: very small n given the effect size claimed (e.g., n=12 claiming a subtle effect)
8. Reproducibility: methods section too vague to allow replication
check_type for all findings must be "methodology".
{_FINDING_SCHEMA}"""

    out = _call_llm(client, model=DEEP_MODEL, max_tokens=1500, prompt=prompt)
    if out is None:
        return []
    result = _extract_json(out)
    return result if isinstance(result, list) else []
