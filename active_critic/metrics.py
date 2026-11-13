from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass(frozen=True)
class CorrelationMetrics:
    pearson: float | None
    spearman: float | None
    kendall: float | None
    pearson_p: float | None
    spearman_p: float | None
    kendall_p: float | None

    @property
    def objective(self) -> float | None:
        values = (self.pearson, self.spearman, self.kendall)
        return None if any(value is None for value in values) else sum(values)

    def require_objective(self, context: str) -> float:
        value = self.objective
        if value is None:
            raise ValueError(
                f"{context}: Q is undefined because a correlation is undefined "
                "(for example, constant predictions or human scores). "
                "No zero-valued correlation or improvement has been substituted."
            )
        return value

    def to_dict(self) -> dict[str, float | None]:
        result = asdict(self)
        result["objective_sum"] = self.objective
        result["average_correlation"] = None if self.objective is None else self.objective / 3
        return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def calculate_correlations(
    predictions: Sequence[float], labels: Sequence[float]
) -> CorrelationMetrics:
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")
    if len(predictions) < 2:
        raise ValueError("At least two examples are required for correlation")
    if any(isinstance(x, (bool, np.bool_)) for x in [*predictions, *labels]):
        raise ValueError("Boolean values are not numerical scores or human ratings")
    predictions, labels = [float(x) for x in predictions], [float(x) for x in labels]
    if not all(math.isfinite(x) for x in [*predictions, *labels]):
        raise ValueError("Correlations require finite scores")
    # A constant vector has undefined correlation, not observed zero correlation.
    # JSON null preserves that distinction; optimization requires a defined Q.
    if len(set(predictions)) == 1 or len(set(labels)) == 1:
        return CorrelationMetrics(None, None, None, None, None, None)
    pearson = pearsonr(predictions, labels)
    spearman = spearmanr(predictions, labels)
    kendall = kendalltau(predictions, labels)
    return CorrelationMetrics(
        pearson=_finite_or_none(pearson.statistic),
        spearman=_finite_or_none(spearman.statistic),
        kendall=_finite_or_none(kendall.statistic),
        pearson_p=_finite_or_none(pearson.pvalue),
        spearman_p=_finite_or_none(spearman.pvalue),
        kendall_p=_finite_or_none(kendall.pvalue),
    )
