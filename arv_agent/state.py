from __future__ import annotations
from typing import TypedDict, Optional, Literal


class Finding(TypedDict):
    check_type: str          # statistical / claim_support / citation / language / methodology
    severity: str            # low / medium / high / critical
    title: str               # short label, max ~8 words
    evidence: str            # exact quote or specific observation
    explanation: str         # why this is a problem


class AgentState(TypedDict):
    # input
    paper_source: str
    # extracted
    paper_text: str
    paper_metadata: dict     # title, authors, year, doi, abstract, source
    claims: list             # list of {claim_text, location, type}
    # planning
    paper_type: str          # empirical / review / meta_analysis / theoretical / unclear
    planned_checks: list     # ordered list of check type strings
    # progress
    checks_completed: list   # check types already run
    findings: list           # accumulated Finding dicts
    errors: list             # error strings
    # output
    final_report: Optional[str]
    status: Literal["running", "done", "error"]
