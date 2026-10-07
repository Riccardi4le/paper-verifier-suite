from itertools import combinations
from typing import Literal
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from cpc_agent.models import AgentState, Contradiction, ConflictingClaim
from cpc_agent.llm import call_json
from cpc_agent.config import PROMPTS_DIR

console = Console()

_P_METHOD = (PROMPTS_DIR / "compare_method.txt").read_text(encoding="utf-8")
_P_POP = (PROMPTS_DIR / "compare_population.txt").read_text(encoding="utf-8")
_P_DEF = (PROMPTS_DIR / "compare_definitions.txt").read_text(encoding="utf-8")
_P_DIAG = (PROMPTS_DIR / "diagnose.txt").read_text(encoding="utf-8")


class _CompareResponse(BaseModel):
    key_differences: list[str]
    could_explain_conflict: bool
    explanation: str


class _MethodResponse(_CompareResponse):
    same_design: bool


class _PopResponse(_CompareResponse):
    populations_comparable: bool


class _DefResponse(BaseModel):
    same_construct: bool
    definitional_differences: list[str]
    could_explain_conflict: bool
    explanation: str


class _DiagResponse(BaseModel):
    diagnosis: Literal["methodological", "population", "definitional", "genuine_conflict", "outdated_data"]
    explanation: str
    recommendation: str


_VALID_DIAGNOSES = {"methodological", "population", "definitional", "genuine_conflict", "outdated_data"}


def _get_method_excerpt(paper_full_text: str) -> str:
    lower = paper_full_text.lower()
    idx = lower.find("method")
    if idx == -1:
        return paper_full_text[:3000]
    return paper_full_text[idx: idx + 3000]


def investigate_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 4 — Investigating Conflicts")
    all_claims = state["all_claims"]
    papers_by_id = {p.id: p for p in state["papers"]}
    contradictions: list[Contradiction] = []
    errors = list(state.get("errors", []))

    flagged = [
        c for c in state["clusters"]
        if state["conflict_verdicts"].get(c.id) in ("conflict", "partial")
    ]

    if not flagged:
        console.print("[yellow]No clusters to investigate.[/yellow]")
        return {**state, "contradictions": []}

    for cluster in track(flagged, description="Investigating...", console=console):
        try:
            verdict = state["conflict_verdicts"][cluster.id]

            claims_in_cluster = [all_claims[i] for i in cluster.claim_indices]
            by_paper: dict[str, list] = {}
            for c in claims_in_cluster:
                by_paper.setdefault(c.paper_id, []).append(c)

            if len(by_paper) < 2:
                continue

            sorted_pids = sorted(by_paper.keys())
            pid_a, pid_b = sorted_pids[0], sorted_pids[1]
            ca = by_paper[pid_a][0]
            cb = by_paper[pid_b][0]
            paper_a = papers_by_id[pid_a]
            paper_b = papers_by_id[pid_b]

            method_result = call_json(
                _P_METHOD.format(
                    topic=cluster.topic,
                    title_a=paper_a.title,
                    method_excerpt_a=_get_method_excerpt(paper_a.full_text),
                    title_b=paper_b.title,
                    method_excerpt_b=_get_method_excerpt(paper_b.full_text),
                ),
                _MethodResponse,
            )

            pop_result = call_json(
                _P_POP.format(
                    topic=cluster.topic,
                    title_a=paper_a.title,
                    claim_a=ca.claim,
                    conditions_a=ca.conditions,
                    n_a=ca.n or "not reported",
                    title_b=paper_b.title,
                    claim_b=cb.claim,
                    conditions_b=cb.conditions,
                    n_b=cb.n or "not reported",
                ),
                _PopResponse,
            )

            def_result = call_json(
                _P_DEF.format(
                    title_a=paper_a.title,
                    claim_a=ca.claim,
                    title_b=paper_b.title,
                    claim_b=cb.claim,
                ),
                _DefResponse,
            )

            claims_summary_lines = []
            for pid, claims in by_paper.items():
                p = papers_by_id[pid]
                for c in claims[:2]:
                    claims_summary_lines.append(f'- [{p.title[:50]}]: "{c.claim}" ({c.conditions})')
            claims_summary = "\n".join(claims_summary_lines)

            diag_result = call_json(
                _P_DIAG.format(
                    topic=cluster.topic,
                    claims_summary=claims_summary,
                    method_comparison=method_result.get("explanation", ""),
                    population_comparison=pop_result.get("explanation", ""),
                    definitions_comparison=def_result.get("explanation", ""),
                ),
                _DiagResponse,
            )

            diagnosis = diag_result["diagnosis"]
            if diagnosis not in _VALID_DIAGNOSES:
                diagnosis = "genuine_conflict"

            conflicting_claims = []
            for pid, claims in by_paper.items():
                p = papers_by_id[pid]
                for c in claims[:2]:
                    conflicting_claims.append(ConflictingClaim(
                        paper_id=pid,
                        paper_title=p.title,
                        claim=c.claim,
                        evidence=c.evidence,
                        conditions=c.conditions,
                    ))

            contradictions.append(Contradiction(
                cluster_id=cluster.id,
                cluster_topic=cluster.topic,
                verdict=verdict,
                conflicting_claims=conflicting_claims,
                diagnosis=diagnosis,
                explanation=diag_result["explanation"],
                recommendation=diag_result["recommendation"],
            ))
            console.print(f"  [red]•[/red] {cluster.topic[:55]} → [{diagnosis}]")

        except Exception as e:
            msg = f"Investigation failed for {cluster.id}: {e}"
            errors.append(msg)
            console.print(f"  [red]✗[/red] {msg}")

    console.print(f"\n[bold]{len(contradictions)} contradiction(s) fully diagnosed.[/bold]")
    return {**state, "contradictions": contradictions, "errors": errors}


def investigate_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 4 - Investigating Conflicts")
    all_claims = state["all_claims"]
    papers_by_id = {paper.id: paper for paper in state["papers"]}
    contradictions: list[Contradiction] = []
    errors = list(state.get("errors", []))

    flagged = [
        cluster for cluster in state["clusters"]
        if state["conflict_verdicts"].get(cluster.id) in ("conflict", "partial")
    ]

    if not flagged:
        console.print("[yellow]No clusters to investigate.[/yellow]")
        return {**state, "contradictions": []}

    for cluster in track(flagged, description="Investigating...", console=console):
        try:
            verdict = state["conflict_verdicts"][cluster.id]

            claims_in_cluster = [all_claims[i] for i in cluster.claim_indices]
            by_paper: dict[str, list] = {}
            for claim in claims_in_cluster:
                by_paper.setdefault(claim.paper_id, []).append(claim)

            if len(by_paper) < 2:
                continue

            claims_summary_lines = []
            for pid, claims in by_paper.items():
                paper = papers_by_id[pid]
                for claim in claims[:2]:
                    claims_summary_lines.append(f'- [{paper.title[:50]}]: "{claim.claim}" ({claim.conditions})')
            claims_summary = "\n".join(claims_summary_lines)

            pair_results = []
            for pid_a, pid_b in combinations(sorted(by_paper.keys()), 2):
                claim_a = by_paper[pid_a][0]
                claim_b = by_paper[pid_b][0]
                paper_a = papers_by_id[pid_a]
                paper_b = papers_by_id[pid_b]

                method_result = call_json(
                    _P_METHOD.format(
                        topic=cluster.topic,
                        title_a=paper_a.title,
                        method_excerpt_a=_get_method_excerpt(paper_a.full_text),
                        title_b=paper_b.title,
                        method_excerpt_b=_get_method_excerpt(paper_b.full_text),
                    ),
                    _MethodResponse,
                )

                pop_result = call_json(
                    _P_POP.format(
                        topic=cluster.topic,
                        title_a=paper_a.title,
                        claim_a=claim_a.claim,
                        conditions_a=claim_a.conditions,
                        n_a=claim_a.n or "not reported",
                        title_b=paper_b.title,
                        claim_b=claim_b.claim,
                        conditions_b=claim_b.conditions,
                        n_b=claim_b.n or "not reported",
                    ),
                    _PopResponse,
                )

                def_result = call_json(
                    _P_DEF.format(
                        title_a=paper_a.title,
                        claim_a=claim_a.claim,
                        title_b=paper_b.title,
                        claim_b=claim_b.claim,
                    ),
                    _DefResponse,
                )

                pair_results.append({
                    "pair": (pid_a, pid_b),
                    "method": method_result,
                    "population": pop_result,
                    "definitions": def_result,
                })

            method_comparison = "\n".join(
                f"{papers_by_id[pid_a].title[:40]} vs {papers_by_id[pid_b].title[:40]}: {result['method'].get('explanation', '')}"
                for result in pair_results
                for pid_a, pid_b in [result["pair"]]
            )
            population_comparison = "\n".join(
                f"{papers_by_id[pid_a].title[:40]} vs {papers_by_id[pid_b].title[:40]}: {result['population'].get('explanation', '')}"
                for result in pair_results
                for pid_a, pid_b in [result["pair"]]
            )
            definitions_comparison = "\n".join(
                f"{papers_by_id[pid_a].title[:40]} vs {papers_by_id[pid_b].title[:40]}: {result['definitions'].get('explanation', '')}"
                for result in pair_results
                for pid_a, pid_b in [result["pair"]]
            )

            diag_result = call_json(
                _P_DIAG.format(
                    topic=cluster.topic,
                    claims_summary=claims_summary,
                    method_comparison=method_comparison,
                    population_comparison=population_comparison,
                    definitions_comparison=definitions_comparison,
                ),
                _DiagResponse,
            )

            diagnosis = diag_result["diagnosis"]
            if diagnosis not in _VALID_DIAGNOSES:
                diagnosis = "genuine_conflict"

            conflicting_claims = []
            for pid, claims in by_paper.items():
                paper = papers_by_id[pid]
                for claim in claims[:2]:
                    conflicting_claims.append(ConflictingClaim(
                        paper_id=pid,
                        paper_title=paper.title,
                        claim=claim.claim,
                        evidence=claim.evidence,
                        conditions=claim.conditions,
                    ))

            contradictions.append(Contradiction(
                cluster_id=cluster.id,
                cluster_topic=cluster.topic,
                verdict=verdict,
                conflicting_claims=conflicting_claims,
                diagnosis=diagnosis,
                explanation=diag_result["explanation"],
                recommendation=diag_result["recommendation"],
            ))
            console.print(f"  [red]-[/red] {cluster.topic[:55]} [{diagnosis}] across {len(by_paper)} paper(s)")

        except Exception as exc:
            msg = f"Investigation failed for {cluster.id}: {exc}"
            errors.append(msg)
            console.print(f"  [red]x[/red] {msg}")

    console.print(f"\n[bold]{len(contradictions)} contradiction(s) fully diagnosed.[/bold]")
    return {**state, "contradictions": contradictions, "errors": errors}
