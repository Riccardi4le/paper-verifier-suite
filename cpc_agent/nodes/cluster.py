import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from rich.console import Console
from cpc_agent.models import AgentState, Cluster
from cpc_agent.tools.embeddings import embed_texts
from cpc_agent.llm import call_text
from cpc_agent.config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES

console = Console()


def _label_cluster(claim_texts: list[str]) -> str:
    joined = "\n".join(f"- {t}" for t in claim_texts[:6])
    prompt = (
        f"These scientific claims all address the same topic:\n{joined}\n\n"
        "Write a short topic label (5-8 words) that captures their shared subject. "
        "Return ONLY the label, no explanation."
    )
    return call_text(prompt)


def cluster_node(state: AgentState) -> AgentState:
    console.rule("[bold blue]Phase 2 — Clustering Claims")
    all_claims = state["all_claims"]
    errors = list(state.get("errors", []))

    if len(all_claims) < 2:
        console.print("[yellow]Not enough claims to cluster.[/yellow]")
        return {**state, "clusters": []}

    claim_texts = [c.claim for c in all_claims]
    embeddings = embed_texts(claim_texts)

    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    non_noise = unique_labels - {-1}
    noise_count = int(np.sum(labels == -1))

    if len(non_noise) == 0:
        console.print("[yellow]HDBSCAN found no clusters — falling back to agglomerative.[/yellow]")
        if len(all_claims) == 2:
            labels = np.zeros(len(all_claims), dtype=int)
        else:
            n_clusters = max(1, min(len(all_claims) - 1, len(all_claims) // 3 or 2))
            agg = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
            labels = agg.fit_predict(embeddings)
        non_noise = set(labels)
        noise_count = 0

    clusters: list[Cluster] = []
    for lbl in sorted(non_noise):
        indices = [i for i, l in enumerate(labels) if l == lbl]
        paper_ids = list({all_claims[i].paper_id for i in indices})
        texts = [claim_texts[i] for i in indices]
        try:
            topic = _label_cluster(texts)
        except Exception as e:
            topic = f"Topic {lbl} ({len(indices)} claims)"
            errors.append(f"Cluster label generation failed for cluster_{lbl}: {e}")
        clusters.append(Cluster(
            id=f"cluster_{lbl}",
            claim_indices=indices,
            topic=topic,
            paper_ids=paper_ids,
        ))
        console.print(f"  [cyan]•[/cyan] {topic[:60]} — {len(indices)} claims from {len(paper_ids)} paper(s)")

    console.print(f"\n[bold]{len(clusters)} clusters found[/bold] ({noise_count} noise claims discarded).")
    return {**state, "clusters": clusters, "errors": errors}
