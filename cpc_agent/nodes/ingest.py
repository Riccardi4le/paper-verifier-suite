from rich.console import Console
from cpc_agent.models import AgentState
from cpc_agent.tools.pdf import extract_paper

console = Console()


def ingest_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 0 — Ingesting PDFs")
    papers = []
    errors = list(state.get("errors", []))

    for path in state["pdf_paths"]:
        try:
            paper = extract_paper(path)
            papers.append(paper)
            console.print(f"  [green]✓[/green] {paper.title[:80]} ({len(paper.full_text):,} chars)")
        except Exception as e:
            msg = f"Failed to ingest {path}: {e}"
            errors.append(msg)
            console.print(f"  [red]✗[/red] {msg}")

    if len(papers) < 2:
        raise RuntimeError(
            f"Only {len(papers)} paper(s) could be ingested — need at least 2 to detect contradictions. Aborting."
        )

    console.print(f"\n[bold]{len(papers)} paper(s) loaded.[/bold]")
    return {**state, "papers": papers, "errors": errors}
