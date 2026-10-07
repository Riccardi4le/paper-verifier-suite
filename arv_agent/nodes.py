"""LangGraph node functions for the Adversarial Reviewer Agent."""

from __future__ import annotations
from .state import AgentState
from .pdf import ingest_paper
from .grim import grim_test, extract_means_and_ns, extract_pvalues, pvalue_cluster_analysis
from .llm import classify_paper, extract_claims, run_check

_CHECKS_BY_TYPE: dict[str, list[str]] = {
    "empirical":     ["statistical", "claim_support", "citation", "language", "methodology"],
    "review":        ["claim_support", "citation", "language", "methodology"],
    "meta_analysis": ["statistical", "claim_support", "citation", "methodology"],
    "theoretical":   ["claim_support", "language", "methodology"],
    "unclear":       ["statistical", "claim_support", "citation", "language", "methodology"],
}

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ── Nodes ─────────────────────────────────────────────────────────────────────

def ingest_node(state: AgentState) -> dict:
    try:
        text, metadata = ingest_paper(state["paper_source"])
    except Exception as exc:
        return {
            "paper_text": "",
            "paper_metadata": {},
            "claims": [],
            "planned_checks": [],
            "checks_completed": [],
            "findings": [],
            "errors": [f"Ingestion failed: {exc}"],
            "status": "error",
        }

    if not text or len(text.strip()) < 200:
        return {
            "paper_text": text or "",
            "paper_metadata": metadata or {},
            "claims": [],
            "planned_checks": [],
            "checks_completed": [],
            "findings": [],
            "errors": [
                "Extracted text is empty or too short (<200 chars). "
                "The PDF may be scanned/image-only (OCR required), "
                "or the URL did not return readable content."
            ],
            "status": "error",
        }

    return {
        "paper_text": text,
        "paper_metadata": metadata,
        "claims": [],
        "planned_checks": [],
        "checks_completed": [],
        "findings": [],
        "errors": [],
        "status": "running",
    }


def classify_node(state: AgentState) -> dict:
    if state.get("status") == "error":
        return {}
    result = classify_paper(state["paper_text"], state["paper_metadata"])
    return {"paper_type": result.get("paper_type", "unclear")}


def plan_node(state: AgentState) -> dict:
    if state.get("status") == "error":
        return {}
    paper_type = state.get("paper_type", "unclear")
    checks = list(_CHECKS_BY_TYPE.get(paper_type, _CHECKS_BY_TYPE["unclear"]))
    claims = extract_claims(state["paper_text"], state["paper_metadata"])
    return {"planned_checks": checks, "claims": claims}


def check_node(state: AgentState) -> dict:
    if state.get("status") == "error":
        return {}

    completed = list(state.get("checks_completed", []))
    planned = state.get("planned_checks", [])
    pending = [c for c in planned if c not in completed]

    if not pending:
        return {"status": "done"}

    check_type = pending[0]
    grim_info = None
    pvalue_info = None
    if check_type == "statistical":
        text = state["paper_text"]
        pairs = extract_means_and_ns(text)
        bad = []
        for mean_text, n in pairs:
            result = grim_test(mean_text, n)
            if not result.consistent:
                bad.append(f"M={mean_text} n={n} (nearest valid M={result.expected:.4f})")
        grim_info = {
            "inconsistent": len(bad),
            "detail": "; ".join(bad) if bad else "No GRIM failures detected by pre-scan",
        }
        pvalue_info = pvalue_cluster_analysis(extract_pvalues(text))

    errors = list(state.get("errors", []))
    try:
        new_findings = run_check(
            check_type=check_type,
            paper_text=state["paper_text"],
            metadata=state["paper_metadata"],
            claims=state["claims"],
            grim_info=grim_info,
            pvalue_info=pvalue_info,
        )
    except Exception as exc:
        new_findings = []
        errors.append(f"Check '{check_type}' failed: {exc}")

    findings = list(state.get("findings", [])) + (new_findings or [])
    completed.append(check_type)
    return {"findings": findings, "checks_completed": completed, "errors": errors}


def report_node(state: AgentState) -> dict:
    findings = state.get("findings", [])
    metadata = state.get("paper_metadata", {})

    title = (metadata.get("title") or "Unknown Paper").strip() or "Unknown Paper"
    year = metadata.get("year", "?")
    doi = metadata.get("doi")
    paper_type = state.get("paper_type", "unclear")
    checks_run = state.get("checks_completed", [])

    counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for finding in findings:
        severity = finding.get("severity", "low")
        if severity in counts:
            counts[severity] += 1

    run_failed = state.get("status") == "error"
    incomplete = not checks_run or len(checks_run) < len(state.get("planned_checks", []))
    has_runtime_errors = bool(state.get("errors"))

    if run_failed or incomplete:
        rec_icon, recommendation = "WARNING", "**Analysis incomplete.** Do not treat this as a clean review result."
    elif has_runtime_errors:
        rec_icon, recommendation = "CAUTION", "**Use with caution.** Some checks failed, so coverage is incomplete."
    elif counts["critical"] >= 2 or counts["critical"] + counts["high"] >= 4:
        rec_icon, recommendation = "BLOCK", "**DO NOT CITE** without independent external verification."
    elif counts["critical"] >= 1 or counts["high"] >= 2:
        rec_icon, recommendation = "HIGH-RISK", "**Cite with caution.** Significant issues detected."
    elif counts["medium"] >= 3:
        rec_icon, recommendation = "CAUTION", "**Use with caveats.** Moderate issues detected."
    else:
        rec_icon, recommendation = "OK", "**Appears acceptable.** Minor or no issues found."

    sorted_findings = sorted(
        findings,
        key=lambda finding: _SEVERITY_RANK.get(finding.get("severity", "low"), 3),
    )

    lines: list[str] = []
    lines.append(f"# Adversarial Report - {title} ({year})\n")
    if doi:
        lines.append(f"**DOI:** {doi}  ")
    lines.append(f"**Paper type:** {paper_type}  ")
    lines.append(f"**Checks run:** {', '.join(checks_run) or 'none'}\n")
    lines.append("---\n")

    lines.append("## Summary\n")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for severity in ("critical", "high", "medium", "low"):
        lines.append(f"| **{severity.upper()}** | {counts[severity]} |")
    lines.append(f"\n{rec_icon} **Recommendation:** {recommendation}\n")
    lines.append("---\n")

    for severity in ("critical", "high", "medium", "low"):
        group = [finding for finding in sorted_findings if finding.get("severity") == severity]
        if not group:
            continue
        lines.append(f"## {severity.upper()} ({len(group)})\n")
        for i, finding in enumerate(group, 1):
            label = f"[{severity[0].upper()}{i}]"
            lines.append(f"### {label} {finding.get('title', 'Untitled finding')}\n")
            if finding.get("evidence"):
                lines.append(f"> {finding['evidence']}\n")
            if finding.get("explanation"):
                lines.append(f"{finding['explanation']}\n")

    if state.get("errors"):
        lines.append("## Errors\n")
        for err in state["errors"]:
            lines.append(f"- {err}")

    return {"final_report": "\n".join(lines), "status": "done"}
