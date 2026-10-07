"""
Pure-Python statistical sanity checks.

GRIM (Granularity-Related Inconsistency of Means):
  For integer-scored items with n respondents, mean*n must be a whole number.
  If the nearest integer / n != mean (within rounding tolerance), the value
  is arithmetically impossible and likely indicates data fabrication or error.

P-value cluster detection:
  A suspicious proportion of p-values falling in [0.040, 0.050] suggests
  optional stopping, selective reporting, or p-hacking.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


@dataclass
class GrimResult:
    mean: float
    n: int
    consistent: bool
    expected: float  # nearest arithmetically valid mean


def _mean_decimal_places(raw_mean: str) -> int:
    if "." not in raw_mean:
        return 0
    return len(raw_mean.split(".", 1)[1])


def grim_test(mean: float | str, n: int) -> GrimResult:
    raw_mean = str(mean).strip()
    decimals = _mean_decimal_places(raw_mean)

    try:
        mean_decimal = Decimal(raw_mean)
    except InvalidOperation:
        mean_decimal = Decimal(str(float(mean)))

    quantum = Decimal("1").scaleb(-decimals)
    approx_sum = mean_decimal * n
    nearest_sum = int(approx_sum.to_integral_value(rounding=ROUND_HALF_UP))
    candidate_sums = range(nearest_sum - 2, nearest_sum + 3)

    consistent = False
    matching_expected: Decimal | None = None
    for candidate_sum in candidate_sums:
        candidate_mean = Decimal(candidate_sum) / Decimal(n)
        rounded = candidate_mean.quantize(quantum, rounding=ROUND_HALF_UP)
        if rounded == mean_decimal:
            consistent = True
            matching_expected = candidate_mean
            break

    expected = matching_expected if matching_expected is not None else Decimal(nearest_sum) / Decimal(n)
    return GrimResult(mean=float(mean_decimal), n=n, consistent=consistent, expected=float(expected))


def extract_means_and_ns(text: str) -> list[tuple[str, int]]:
    """Heuristically extract (raw_mean_text, n) pairs from paper text."""
    pairs: list[tuple[str, int]] = []

    # Matches patterns like: M = 3.47 ... n = 82  (within ~200 chars)
    for m in re.finditer(r'[Mm](?:ean)?\s*=\s*(\d+\.\d+)', text):
        snippet = text[m.start(): m.start() + 250]
        n_match = re.search(r'\bn\s*=\s*(\d+)', snippet)
        if n_match:
            try:
                mean_text = m.group(1)
                mean = float(mean_text)
                n = int(n_match.group(1))
                if 1.0 < mean < 100.0 and 2 <= n <= 10_000:
                    pairs.append((mean_text, n))
            except ValueError:
                pass

    return pairs[:20]


def extract_pvalues(text: str) -> list[float]:
    """Extract all reported p-values from paper text."""
    values: list[float] = []
    pattern = re.compile(
        r'\bp\s*[=<>≤≥]\s*(0?\.0*[0-9]+(?:e[-−]?\d+)?)',
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        try:
            v = float(m.group(1))
            if 0 < v <= 1:
                values.append(v)
        except ValueError:
            pass
    return values


def pvalue_cluster_analysis(pvalues: list[float]) -> dict:
    """
    Returns:
      total, near_threshold (p in [0.040, 0.050]), suspicious (bool), detail (str)
    """
    if not pvalues:
        return {
            "total": 0,
            "near_threshold": 0,
            "suspicious": False,
            "detail": "No p-values found in text",
        }

    near = [p for p in pvalues if 0.040 <= p <= 0.050]
    ratio = len(near) / len(pvalues)
    # Suspicious if ≥5 p-values and ≥40% cluster near threshold (expected ~10% by chance)
    suspicious = len(pvalues) >= 5 and ratio >= 0.40

    detail = (
        f"{len(near)}/{len(pvalues)} p-values in [0.040, 0.050] "
        f"({ratio * 100:.0f}% — expected ~10% by chance)"
    )
    return {
        "total": len(pvalues),
        "near_threshold": len(near),
        "suspicious": suspicious,
        "detail": detail,
    }
