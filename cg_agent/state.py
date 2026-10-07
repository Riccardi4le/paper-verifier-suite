from __future__ import annotations
from typing import TypedDict, Optional, Literal

Verdict = Literal["supported", "partial", "distorted", "contradicted", "not_found"]


class PaperData(TypedDict, total=False):
    paper_id: str
    openalex_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]
    source: str
    referenced_work_ids: list[str]


class TreeNode(TypedDict, total=False):
    node_id: str
    paper: Optional[PaperData]
    claim_text: Optional[str]
    verdict: Optional[Verdict]
    reasoning: Optional[str]
    is_primary: bool
    evidence_type: str
    children: list[str]
    parent_id: Optional[str]
    depth: int
    # temp fields used between fetch → analyze (prefixed _)
    _parent_id: Optional[str]
    _depth: int


class QueueItem(TypedDict):
    node_id: str
    ref_query: str
    depth: int
    parent_id: Optional[str]


class AgentState(TypedDict):
    # input
    original_claim: str
    source_ref: str
    max_depth: int
    # working
    tree: dict                   # node_id → TreeNode
    queue: list                  # list[QueueItem]
    visited: list                # list[str] paper_ids
    errors: list                 # list[str]
    current_node_id: Optional[str]
    # output
    final_report: Optional[str]
    status: Literal["running", "done", "error"]
