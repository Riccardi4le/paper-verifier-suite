from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Claim(BaseModel):
    claim: str = Field(description="The central scientific assertion")
    evidence: str = Field(description="Supporting data or statistic")
    conditions: str = Field(description="Context or caveats (e.g. 'in motivated subjects')")
    n: int | None = Field(default=None, description="Sample size if present")
    method: str = Field(description="Study design in one line")
    paper_id: str = Field(description="ID of the source paper")


class ClaimList(BaseModel):
    claims: list[Claim]


class Paper(BaseModel):
    id: str
    path: str
    title: str
    authors: str
    full_text: str


class Cluster(BaseModel):
    id: str
    claim_indices: list[int]
    topic: str
    paper_ids: list[str]


class ConflictingClaim(BaseModel):
    paper_id: str
    paper_title: str
    claim: str
    evidence: str
    conditions: str


class Contradiction(BaseModel):
    cluster_id: str
    cluster_topic: str
    verdict: Literal["conflict", "partial"]
    conflicting_claims: list[ConflictingClaim]
    diagnosis: Literal[
        "methodological", "population", "definitional",
        "genuine_conflict", "outdated_data"
    ]
    explanation: str
    recommendation: str


class AgentState(TypedDict):
    pdf_paths: list[str]
    topic: str
    papers: list[Paper]
    all_claims: list[Claim]
    claims_per_paper: dict[str, list[int]]
    clusters: list[Cluster]
    conflict_verdicts: dict[str, Literal["agreement", "partial", "conflict", "unrelated"]]
    contradictions: list[Contradiction]
    output_path: str
    errors: list[str]
