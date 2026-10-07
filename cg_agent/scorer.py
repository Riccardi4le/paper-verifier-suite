"""Distortion score computation from the citation tree."""

from __future__ import annotations

_WEIGHTS: dict[str, float] = {
    "supported": 0.0,
    "partial": 0.3,
    "distorted": 0.7,
    "contradicted": 1.0,
    "not_found": 0.4,
}


def compute_distortion_score(tree: dict) -> float:
    """
    Score 0.0 (no distortion) → 1.0 (maximum distortion).
    Deeper hops are weighted slightly more because distortions compound.
    """
    scores: list[float] = []
    max_depth = max((n.get("depth", 0) for n in tree.values()), default=0)

    for node in tree.values():
        verdict = node.get("verdict")
        if verdict not in _WEIGHTS:
            continue
        depth = node.get("depth", 0)
        # linear depth weight: deeper = up to 2x
        weight = 1.0 + (depth / max(max_depth, 1))
        scores.append(_WEIGHTS[verdict] * weight)

    if not scores:
        return 0.0

    raw = sum(scores) / len(scores)
    # Normalise back to 0-1 (max possible weight per verdict = 1.0*2 = 2.0 but
    # _WEIGHTS["contradicted"]=1.0 so max raw ≈ 2.0)
    normalised = min(raw / 2.0, 1.0)
    return round(normalised, 2)
