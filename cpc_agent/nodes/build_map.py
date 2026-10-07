from datetime import datetime
from pathlib import Path
from rich.console import Console
from cpc_agent.models import AgentState, Contradiction
from cpc_agent.config import OUTPUT_DIR

console = Console()

_VERDICT_ICONS = {
    "conflict": "🔴",
    "partial": "🟡",
    "agreement": "🟢",
    "unrelated": "⚪",
}

_DIAGNOSIS_LABELS = {
    "methodological": "Methodological — different study designs or metrics",
    "population": "Population — different study populations",
    "definitional": "Definitional — same terms, different meanings",
    "genuine_conflict": "Genuine Conflict — same methods/population, contradictory results",
    "outdated_data": "Outdated Data — newer evidence supersedes older finding",
}


def _md_cell(text: str, limit: int) -> str:
    """Sanitize a string for inclusion in a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()[:limit]


def _render_contradiction(contra: Contradiction, idx: int) -> str:
    icon = _VERDICT_ICONS.get(contra.verdict, "🔴")
    lines = [
        f"## Cluster {idx} — {contra.cluster_topic}",
        "",
        f"{icon} **{contra.verdict.upper()}**",
        "",
        "### Conflicting Claims",
        "",
    ]
    seen_papers: set[str] = set()
    for cc in contra.conflicting_claims:
        if cc.paper_id in seen_papers:
            continue
        seen_papers.add(cc.paper_id)
        lines += [
            f"**{cc.paper_title[:80]}**",
            f"> \"{cc.claim}\"",
            f"- Evidence: {cc.evidence}",
            f"- Conditions: {cc.conditions}",
            "",
        ]

    diag_label = _DIAGNOSIS_LABELS.get(contra.diagnosis, contra.diagnosis)
    lines += [
        f"### Diagnosis: `{contra.diagnosis}`",
        f"_{diag_label}_",
        "",
        contra.explanation,
        "",
        f"### Recommendation",
        "",
        contra.recommendation,
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _render_agreement_cluster(cluster_topic: str, paper_count: int, idx: int) -> str:
    return "\n".join([
        f"## Cluster {idx} — {cluster_topic}",
        "",
        f"🟢 **AGREEMENT** — {paper_count} paper(s) converge on this topic.",
        "",
        "---",
        "",
    ])


def build_map_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 5 — Building Contradiction Map")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"contradiction_map_{ts}.md"

    papers = state["papers"]
    clusters = state["clusters"]
    contradictions_by_cluster = {c.cluster_id: c for c in state["contradictions"]}
    verdicts = state["conflict_verdicts"]

    sorted_clusters = sorted(
        clusters,
        key=lambda c: {"conflict": 0, "partial": 1, "agreement": 2, "unrelated": 3}.get(
            verdicts.get(c.id, "unrelated"), 3
        ),
    )

    header = [
        f"# Contradiction Map — {state.get('topic', 'Research Topic')}",
        "",
        f"**Papers analyzed:** {len(papers)}  ",
        f"**Total claims extracted:** {len(state.get('all_claims', []))}  ",
        f"**Clusters found:** {len(clusters)}  ",
        f"**Contradictions diagnosed:** {len(state['contradictions'])}  ",
        "",
        "| Paper | Title | Authors |",
        "|-------|-------|---------|",
    ]
    for p in papers:
        header.append(f"| `{p.id}` | {_md_cell(p.title, 60)} | {_md_cell(p.authors, 40)} |")
    header += ["", "---", ""]

    body_sections = []
    cluster_counter = 1
    for cluster in sorted_clusters:
        verdict = verdicts.get(cluster.id, "unrelated")
        if verdict in ("conflict", "partial") and cluster.id in contradictions_by_cluster:
            body_sections.append(
                _render_contradiction(contradictions_by_cluster[cluster.id], cluster_counter)
            )
        elif verdict == "agreement":
            body_sections.append(
                _render_agreement_cluster(cluster.topic, len(cluster.paper_ids), cluster_counter)
            )
        cluster_counter += 1

    if state.get("errors"):
        body_sections.append("## ⚠️ Errors During Processing\n")
        for err in state["errors"]:
            body_sections.append(f"- {err}\n")

    content = "\n".join(header) + "\n" + "".join(body_sections)
    out_path.write_text(content, encoding="utf-8")
    console.print(f"\n[bold green]✓ Contradiction Map saved:[/bold green] {out_path}")

    return {**state, "output_path": str(out_path)}
