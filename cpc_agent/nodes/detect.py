from typing import Literal
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from cpc_agent.models import AgentState
from cpc_agent.llm import call_json
from cpc_agent.config import PROMPTS_DIR

console = Console()
_PROMPT = (PROMPTS_DIR / "detect_conflict.txt").read_text(encoding="utf-8")

_VERDICT_LABELS = {
    "agreement": "[green]🟢 Agreement[/green]",
    "partial": "[yellow]🟡 Partial[/yellow]",
    "conflict": "[red]🔴 Conflict[/red]",
    "unrelated": "[dim]⚪ Unrelated[/dim]",
}


class _VerdictResponse(BaseModel):
    verdict: Literal["agreement", "partial", "conflict", "unrelated"]
    reasoning: str


def detect_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 3 — Detecting Conflicts")
    all_claims = state["all_claims"]
    verdicts: dict[str, str] = {}
    errors = list(state.get("errors", []))

    eligible = [
        c for c in state["clusters"]
        if len(c.paper_ids) >= 2
    ]
    skipped = len(state["clusters"]) - len(eligible)
    if skipped:
        console.print(f"  [dim]{skipped} cluster(s) skipped (single-paper).[/dim]")

    for cluster in track(eligible, description="Detecting conflicts...", console=console):
        try:
            claims_lines = []
            for idx in cluster.claim_indices:
                c = all_claims[idx]
                claims_lines.append(
                    f"[Paper {c.paper_id}] {c.claim} | Evidence: {c.evidence} | Conditions: {c.conditions}"
                )
            claims_text = "\n".join(claims_lines)
            prompt = _PROMPT.format(topic=cluster.topic, claims_text=claims_text)
            result = call_json(prompt, _VerdictResponse)
            verdict = result["verdict"]
            if verdict not in ("agreement", "partial", "conflict", "unrelated"):
                verdict = "unrelated"
            verdicts[cluster.id] = verdict
            label = _VERDICT_LABELS.get(verdict, verdict)
            console.print(f"  {label}: {cluster.topic[:60]}")
        except Exception as e:
            msg = f"Conflict detection failed for {cluster.id}: {e}"
            errors.append(msg)
            verdicts[cluster.id] = "unrelated"
            console.print(f"  [red]✗[/red] {msg}")

    conflict_count = sum(1 for v in verdicts.values() if v in ("conflict", "partial"))
    console.print(f"\n[bold]{conflict_count} cluster(s) flagged for investigation.[/bold]")
    return {**state, "conflict_verdicts": verdicts, "errors": errors}
