from rich.console import Console
from rich.progress import track
from cpc_agent.models import AgentState, Claim, ClaimList
from cpc_agent.llm import call_json
from cpc_agent.config import MAX_CLAIMS_PER_PAPER, MAX_PAPER_CHARS, PROMPTS_DIR

console = Console()
_PROMPT = (PROMPTS_DIR / "extract_claims.txt").read_text(encoding="utf-8")


def _section_priority_text(full_text: str) -> str:
    """Prefer abstract + results + discussion sections, capped at MAX_PAPER_CHARS."""
    lower = full_text.lower()
    sections = []
    for keyword in ["abstract", "results", "discussion", "conclusion"]:
        idx = lower.find(keyword)
        if idx != -1:
            sections.append(full_text[idx: idx + 5000])
    text = "\n\n".join(sections) if sections else full_text
    if len(text) > MAX_PAPER_CHARS:
        text = text[:MAX_PAPER_CHARS]
    return text


def extract_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 1 — Extracting Claims")
    all_claims: list[Claim] = []
    claims_per_paper: dict[str, list[int]] = {}
    errors = list(state.get("errors", []))

    for paper in track(state["papers"], description="Extracting claims...", console=console):
        try:
            text = _section_priority_text(paper.full_text)
            prompt = _PROMPT.format(
                max_claims=MAX_CLAIMS_PER_PAPER,
                paper_id=paper.id,
                paper_text=text,
            )
            result = call_json(prompt, ClaimList)
            raw_claims = result["claims"]

            start_idx = len(all_claims)
            for c in raw_claims:
                c["paper_id"] = paper.id
                all_claims.append(Claim(**c))

            indices = list(range(start_idx, len(all_claims)))
            claims_per_paper[paper.id] = indices
            console.print(f"  [green]✓[/green] {paper.title[:60]} → {len(indices)} claims")
        except Exception as e:
            msg = f"Claim extraction failed for {paper.title[:60]}: {e}"
            errors.append(msg)
            console.print(f"  [red]✗[/red] {msg}")
            claims_per_paper[paper.id] = []

    console.print(f"\n[bold]{len(all_claims)} total claims extracted.[/bold]")
    return {**state, "all_claims": all_claims, "claims_per_paper": claims_per_paper, "errors": errors}
