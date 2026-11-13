"""Instance-level correlations, confidence intervals, and paired significance tests."""
from __future__ import annotations

import statistics as stdlib_statistics
from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

# Correlations are dimensionless and bounded; this tolerance handles floating
# point ties, not an experimental effect-size or significance threshold.
_CORRELATION_ROUNDOFF = 1e-12

def _vectors(*values: Sequence[float]) -> list[np.ndarray]:
    if any(isinstance(x, (bool, np.bool_)) for vector in values for x in vector):
        raise ValueError("Boolean values are not numerical scores or human ratings")
    arrays = [np.asarray(x, dtype=float) for x in values]
    if not arrays or any(x.ndim != 1 or len(x) < 2 for x in arrays):
        raise ValueError("Expected one-dimensional vectors of at least two scores")
    if len({len(x) for x in arrays}) != 1 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Vectors must have equal length and finite scores")
    return arrays


def pearson_bootstrap_ci(
    predictions: Sequence[float], labels: Sequence[float], *,
    resamples: int = 1000, seed: int = 42, confidence: float = 0.95,
) -> dict:
    prediction, label = _vectors(predictions, labels)
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indices = rng.integers(0, len(prediction), len(prediction))
        x, y = prediction[indices], label[indices]
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        values.append(float(pearsonr(x, y).statistic))
    alpha = (1 - confidence) / 2
    interval = np.quantile(values, [alpha, 1 - alpha]).tolist() if values else [None, None]
    return {
        "method": "paired_instance_percentile_bootstrap", "confidence": confidence,
        "resamples": resamples, "valid_resamples": len(values), "seed": seed,
        "lower": interval[0], "upper": interval[1],
    }


def paired_perm_both(
    candidate: Sequence[float], baseline: Sequence[float], labels: Sequence[float], *,
    resamples: int = 1000, seed: int = 42,
) -> dict:
    """PERM-BOTH for pooled instance correlations, one-sided candidate > baseline.

    Separately standardize the two metric vectors, then swap their paired entries
    independently. This follows mt-metrics-eval PermutationSigDiff (average_by=none).
    Run all 1,000 draws; do not substitute correlation-vs-zero analytical p-values.
    """
    candidate, baseline, labels = _vectors(candidate, baseline, labels)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if any(np.ptp(x) == 0 for x in (candidate, baseline, labels)):
        raise ValueError("PERM-BOTH requires nonconstant scores and labels")
    candidate = (candidate - candidate.mean()) / candidate.std()
    baseline = (baseline - baseline.mean()) / baseline.std()
    funcs = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}
    def corr(f, x):
        return 0.0 if np.ptp(x) == 0 else float(f(x, labels).statistic)
    observed = {k: corr(f, candidate) - corr(f, baseline) for k, f in funcs.items()}
    observed = {k: 0.0 if abs(value) <= _CORRELATION_ROUNDOFF else value
                for k, value in observed.items()}
    counts = dict.fromkeys(funcs, 0)
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        swap = rng.integers(0, 2, len(labels)).astype(bool)
        a, b = np.where(swap, baseline, candidate), np.where(swap, candidate, baseline)
        for name, f in funcs.items():
            if corr(f, a) - corr(f, b) >= observed[name] - _CORRELATION_ROUNDOFF:
                counts[name] += 1
    return {
        "method": "paired_PERM_BOTH", "alternative": "candidate_greater",
        "resamples": resamples, "seed": seed,
        "numerical_tie_tolerance": _CORRELATION_ROUNDOFF,
        "metrics": {
            k: {"difference": observed[k], "p_value": counts[k] / resamples,
                "significant_0.05": counts[k] / resamples < 0.05 and observed[k] > 0}
            for k in funcs
        },
    }


def summarize_runs(metrics: Sequence[dict]) -> dict:
    if not metrics:
        raise ValueError("No runs to summarize")
    rows = []
    for metric in metrics:
        values = [metric[k] for k in ("pearson", "spearman", "kendall")]
        average = None if any(value is None for value in values) else sum(values) / 3
        rows.append({**metric, "average_correlation": average})
    summary = {}
    for key in ("pearson", "spearman", "kendall", "average_correlation"):
        values = [m[key] for m in rows if m[key] is not None]
        complete = len(values) == len(metrics)
        # Do not turn missing correlations into zeros or silently summarize a
        # smaller set as though all five reported repetitions were valid.
        summary[key] = {
            "mean": stdlib_statistics.mean(values) if complete else None,
            "sd": stdlib_statistics.stdev(values) if complete and len(values) > 1 else None,
            "defined_runs": len(values), "undefined_runs": len(metrics) - len(values),
        }
    return {
        "runs": len(metrics), "sd_ddof": 1,
        "metrics": summary,
    }
