"""LangGraph node functions for the Citation Genealogy Agent."""

from __future__ import annotations
from typing import Optional

from .state import AgentState
from .api import (
    fetch_work_openalex,
    fetch_referenced_works_metadata,
    work_to_paper,
    format_refs_for_llm,
)
from .llm import analyze_paper
from .scorer import compute_distortion_score


# ── helpers ────────────────────────────────────────────────────────────────────

_VERDICT_EMOJI = {
    "supported": "✅",
    "partial": "⚠️",
    "distorted": "🔴",
    "contradicted": "🚫",
    "not_found": "❓",
}


def _short_title(paper: Optional[dict], max_len: int = 45) -> str:
    if not paper:
        return "Unknown"
    t = paper.get("title") or "Unknown"
    return t[:max_len] + "…" if len(t) > max_len else t


def _safe(s: Optional[str]) -> str:
    """Strip characters that break Mermaid node labels."""
    if not s:
        return ""
    return s.replace('"', "'").replace("[", "(").replace("]", ")")


# ── nodes ──────────────────────────────────────────────────────────────────────

def ingest_node(state: AgentState) -> dict:
    root_id = "node_0"
    root: dict = {
        "node_id": root_id,
        "paper": None,
        "claim_text": None,
        "verdict": None,
        "reasoning": None,
        "is_primary": False,
        "evidence_type": "unclear",
        "children": [],
        "parent_id": None,
        "depth": 0,
    }
    queue_item = {
        "node_id": root_id,
        "ref_query": state["source_ref"],
        "depth": 0,
        "parent_id": None,
    }
    return {
        "tree": {root_id: root},
        "queue": [queue_item],
        "visited": [],
        "errors": [],
        "current_node_id": None,
        "status": "running",
    }


def fetch_node(state: AgentState) -> dict:
    queue = list(state["queue"])
    if not queue:
        return {"status": "done"}

    current = queue.pop(0)
    node_id: str = current["node_id"]
    ref_query: str = current["ref_query"]
    depth: int = current["depth"]
    parent_id: Optional[str] = current.get("parent_id")

    work = fetch_work_openalex(ref_query)
    tree = dict(state["tree"])

    if not work:
        tree[node_id] = {
            **tree.get(node_id, {}),
            "verdict": "not_found",
            "reasoning": f"Could not resolve: {ref_query}",
            "depth": depth,
            "parent_id": parent_id,
            "children": [],
        }
        errors = list(state["errors"]) + [f"Paper not found: {ref_query}"]
        return {"queue": queue, "tree": tree, "errors": errors, "current_node_id": None}

    paper = work_to_paper(work)
    paper_id = paper["paper_id"]
    visited = list(state["visited"])

    if paper_id in visited:
        tree[node_id] = {
            **tree.get(node_id, {}),
            "paper": paper,
            "verdict": "not_found",
            "reasoning": "Cycle detected — already visited this paper",
            "depth": depth,
            "parent_id": parent_id,
            "children": [],
        }
        title = paper.get("title") or ref_query
        errors = list(state["errors"]) + [f"Cycle detected at depth {depth}: {title}"]
        return {
            "queue": queue,
            "tree": tree,
            "visited": visited,
            "errors": errors,
            "current_node_id": None,
        }

    visited.append(paper_id)
    tree[node_id] = {
        **tree.get(node_id, {}),
        "paper": paper,
        "depth": depth,
        "parent_id": parent_id,
        "children": [],
        # temp fields consumed by analyze_node
        "_depth": depth,
        "_parent_id": parent_id,
    }

    return {
        "queue": queue,
        "tree": tree,
        "visited": visited,
        "current_node_id": node_id,
    }


def analyze_node(state: AgentState) -> dict:
    node_id = state.get("current_node_id")
    if not node_id:
        return {}

    tree = dict(state["tree"])
    node = dict(tree.get(node_id, {}))

    # If node already has a terminal verdict (cycle/not found in fetch), skip
    if node.get("verdict") is not None:
        return {"current_node_id": None}

    paper = node.get("paper")
    if not paper:
        return {"current_node_id": None}

    depth: int = node.get("_depth", node.get("depth", 0))
    parent_id: Optional[str] = node.get("_parent_id", node.get("parent_id"))

    # Determine the "reference_claim" we're checking against
    if parent_id and tree.get(parent_id, {}).get("claim_text"):
        reference_claim = tree[parent_id]["claim_text"]
    else:
        reference_claim = state["original_claim"]

    # Fetch referenced works for this paper
    ref_ids = paper.get("referenced_work_ids", [])
    referenced_works = fetch_referenced_works_metadata(ref_ids) if ref_ids else []
    refs_text = format_refs_for_llm(referenced_works)

    # Single LLM call for all analysis
    result = analyze_paper(
        original_claim=state["original_claim"],
        reference_claim=reference_claim,
        paper=paper,
        refs_text=refs_text,
    )

    claim_text = result.get("claim_text")
    verdict = result.get("verdict", "not_found")
    reasoning = result.get("reasoning", "")
    is_primary = result.get("is_primary", False)
    evidence_type = result.get("evidence_type", "unclear")
    next_ref_title = result.get("next_ref_title")
    next_ref_year = result.get("next_ref_year")

    # Build child node if we should recurse
    queue = list(state["queue"])
    new_children: list[str] = []

    should_recurse = (
        result.get("claim_found", False)
        and not is_primary
        and depth < state["max_depth"] - 1
        and next_ref_title
        and verdict != "contradicted"
    )

    if should_recurse:
        child_id = f"node_{len(tree)}"
        ref_query = f"{next_ref_title} {next_ref_year or ''}".strip()

        # Try to resolve ref_query against fetched referenced_works for a better query.
        # Skip refs with missing/empty titles — a "" substring would match anything.
        needle = next_ref_title.lower().strip()
        for rw in referenced_works:
            rw_title = (rw.get("title") or "").lower().strip()
            if not rw_title or rw_title == "unknown":
                continue
            if needle in rw_title or rw_title in needle:
                doi = rw.get("doi")
                if doi:
                    ref_query = doi
                elif rw.get("id"):
                    ref_query = rw["id"]
                break

        child_node: dict = {
            "node_id": child_id,
            "paper": None,
            "claim_text": None,
            "verdict": None,
            "reasoning": None,
            "is_primary": False,
            "evidence_type": "unclear",
            "children": [],
            "parent_id": node_id,
            "depth": depth + 1,
        }
        tree[child_id] = child_node
        new_children.append(child_id)
        queue.append({
            "node_id": child_id,
            "ref_query": ref_query,
            "depth": depth + 1,
            "parent_id": node_id,
        })

    # Update current node (strip temp fields)
    tree[node_id] = {
        "node_id": node_id,
        "paper": paper,
        "claim_text": claim_text,
        "verdict": verdict,
        "reasoning": reasoning,
        "is_primary": is_primary,
        "evidence_type": evidence_type,
        "children": new_children,
        "parent_id": parent_id,
        "depth": depth,
    }

    return {"tree": tree, "queue": queue, "current_node_id": None}


def report_node(state: AgentState) -> dict:
    tree = state["tree"]
    score = compute_distortion_score(tree)

    # ── Mermaid tree ─────────────────────────────────────────────────────────
    mermaid_lines = ["```mermaid", "graph TD"]
    for nid, node in tree.items():
        paper = node.get("paper") or {}
        year = paper.get("year", "?")
        title = _safe(_short_title(paper))
        verdict = node.get("verdict") or "pending"
        emoji = _VERDICT_EMOJI.get(verdict, "⬜")
        primary_tag = " ⭐" if node.get("is_primary") else ""
        label = f"{emoji}{primary_tag} {title} ({year})"
        mermaid_lines.append(f'    {nid}["{label}"]')
        for child_id in node.get("children", []):
            mermaid_lines.append(f"    {nid} --> {child_id}")
    mermaid_lines.append("```")

    # ── Claim evolution table ─────────────────────────────────────────────────
    # Traverse from root to leaves (DFS)
    evolution_rows: list[tuple[int, dict]] = []

    def _traverse(nid: str):
        node = tree.get(nid)
        if not node:
            return
        evolution_rows.append((node.get("depth", 0), node))
        for child_id in node.get("children", []):
            _traverse(child_id)

    for nid, node in tree.items():
        if node.get("parent_id") is None:
            _traverse(nid)
            break

    claim_table = ["| Depth | Paper | Claim text | Verdict |", "|-------|-------|------------|---------|"]
    claim_table.append(
        f"| Source | _(user's paper)_ | {_safe(state['original_claim'])} | → |"
    )
    for depth, node in evolution_rows:
        paper = node.get("paper") or {}
        title = _safe(_short_title(paper, 35))
        year = paper.get("year", "?")
        ct = _safe(node.get("claim_text") or "_not found_")
        v = node.get("verdict") or "pending"
        emoji = _VERDICT_EMOJI.get(v, "⬜")
        primary = " ⭐" if node.get("is_primary") else ""
        claim_table.append(f"| {depth} | {title} ({year}){primary} | {ct} | {emoji} `{v}` |")

    # ── Score bar ─────────────────────────────────────────────────────────────
    bar_emoji = "🟢" if score < 0.3 else ("🟡" if score < 0.6 else "🔴")
    label_text = "low" if score < 0.3 else ("medium" if score < 0.6 else "HIGH"  )

    # ── Assemble report ───────────────────────────────────────────────────────
    sections: list[str] = []
    sections.append("# Citation Genealogy Report\n")
    sections.append(f'**Claim:** "{state["original_claim"]}"  ')
    sections.append(f'**Source:** {state["source_ref"]}  ')
    sections.append(f'**Max depth:** {state["max_depth"]}\n')
    sections.append("---\n")

    sections.append("## Overall Assessment\n")
    sections.append(f"| Metric | Value |")
    sections.append(f"|--------|-------|")
    sections.append(f"| Distortion score | {bar_emoji} **{score:.2f}** / 1.0 ({label_text}) |")
    sections.append(f"| Papers traced | {len(tree)} |")
    primary_nodes = [n for n in tree.values() if n.get("is_primary")]
    if primary_nodes:
        p = primary_nodes[-1].get("paper") or {}
        sections.append(
            f"| Primary source | {_short_title(p)} ({p.get('year', '?')}) |"
        )
    sections.append("")

    sections.append("## Citation Tree\n")
    sections.extend(mermaid_lines)
    sections.append("")

    sections.append("## Claim Evolution\n")
    sections.extend(claim_table)
    sections.append("")

    sections.append("## Node Details\n")
    for nid, node in tree.items():
        paper = node.get("paper") or {}
        title = paper.get("title") or "Unknown"
        year = paper.get("year", "?")
        verdict = node.get("verdict") or "pending"
        emoji = _VERDICT_EMOJI.get(verdict, "⬜")
        primary_str = " ⭐ PRIMARY SOURCE" if node.get("is_primary") else ""
        sections.append(f"### {emoji} {title} ({year}){primary_str}\n")
        if node.get("claim_text"):
            sections.append(f"> {node['claim_text']}\n")
        if verdict != "pending":
            sections.append(f"**Verdict:** `{verdict}`  ")
        if node.get("reasoning"):
            sections.append(f"**Reasoning:** {node['reasoning']}  ")
        if node.get("evidence_type"):
            sections.append(f"**Evidence type:** {node['evidence_type']}")
        if paper.get("doi"):
            sections.append(f"**DOI:** {paper['doi']}")
        sections.append("")

    if state.get("errors"):
        sections.append("## Warnings\n")
        for err in state["errors"]:
            sections.append(f"- {err}")

    final_report = "\n".join(sections)
    return {"final_report": final_report, "status": "done"}
