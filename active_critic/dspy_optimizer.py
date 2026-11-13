from __future__ import annotations

import math
import random
from typing import Any, Sequence

from .metrics import CorrelationMetrics, calculate_correlations
from .parsing import clamp_score


class CorrelationGuidedOptimizer:
    """Correlation-guided demonstration search using DSPy scoring programs.

    Proxy: leave-one-out marginal contributions and candidate ranking.
    Target: initial subset and ONE best proposed replacement per round.
    Q is computed over the entire labeled training set, never test examples.
    """

    def __init__(
        self, *, demo_count: int, proxy_lm: Any,
        epsilon: float = 0.01, max_iterations: int = 11, seed: int = 42,
    ) -> None:
        if demo_count <= 0 or max_iterations <= 0:
            raise ValueError("demo_count and max_iterations must be positive")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if proxy_lm is None:
            raise ValueError("A proxy LM must be supplied explicitly")
        self.demo_count, self.proxy_lm = demo_count, proxy_lm
        self.epsilon, self.max_iterations, self.seed = epsilon, max_iterations, seed
        self.history: list[dict] = []
        self.selected_indices: list[int] = []
        self.target_evaluations = 0
        self.proxy_evaluations = 0
        self.stop_reason = ""
        self._cache: dict[tuple[str, tuple[int, ...]], CorrelationMetrics] = {}

    def _program(self, student: Any, selected: Sequence[int], train: Sequence[Any], *, proxy: bool = False) -> Any:
        program = student.deepcopy()
        program.set_demonstrations([train[i] for i in selected])
        if proxy:
            # context(lm=proxy) alone does NOT override predictor-bound target LMs.
            program.set_lm(self.proxy_lm)
        return program

    def _score(self, student: Any, selected: Sequence[int], train: Sequence[Any], *, proxy: bool) -> CorrelationMetrics:
        key = ("proxy" if proxy else "target", tuple(selected))
        if key not in self._cache:
            program = self._program(student, selected, train, proxy=proxy)
            predictions, labels = [], []
            for example in train:
                prediction = program(**example.inputs())
                predictions.append(clamp_score(prediction.overall_score))
                labels.append(clamp_score(example.score_of_overall))
            self._cache[key] = calculate_correlations(predictions, labels)
            if proxy:
                self.proxy_evaluations += 1
            else:
                self.target_evaluations += 1
        return self._cache[key]

    def compile(self, student: Any, *, trainset: Sequence[Any], initial_indices: Sequence[int] | None = None) -> Any:
        train = list(trainset)
        if len(train) < max(2, self.demo_count):
            raise ValueError("Not enough training examples for this demo budget")
        if len({str(x.uid) for x in train}) != len(train):
            raise ValueError("Training UIDs must be unique")
        if len({clamp_score(x.score_of_overall) for x in train}) < 2:
            raise ValueError("Correlation-guided optimization requires nonconstant human scores")
        self.history, self._cache = [], {}
        self.target_evaluations = self.proxy_evaluations = 0
        selected = (
            list(initial_indices) if initial_indices is not None
            else random.Random(self.seed).sample(range(len(train)), self.demo_count)
        )
        if len(selected) != self.demo_count or len(set(selected)) != len(selected) or any(i < 0 or i >= len(train) for i in selected):
            raise ValueError("Invalid initial demonstration subset")
        current = self._score(student, selected, train, proxy=False)
        current.require_objective("Initial target evaluation")
        self.history.append({
            "round": 0, "selected_ids": [str(train[i].uid) for i in selected],
            "target_metrics": current.to_dict(),
        })
        self.stop_reason = "max_iterations"
        for iteration in range(1, self.max_iterations + 1):
            candidates = [i for i in range(len(train)) if i not in selected]
            if not candidates:
                self.stop_reason = "no_candidates"
                break
            # The formula uses the current target Q as its common baseline;
            # the proxy supplies Q_-i and candidate estimates.
            baseline_q = current.require_objective("Current target evaluation")
            contributions = [
                baseline_q - self._score(student, selected[:p] + selected[p + 1:], train, proxy=True).require_objective("Proxy leave-one-out evaluation")
                for p in range(len(selected))
            ]
            weakest = min(range(len(selected)), key=contributions.__getitem__)
            proposals = []
            for candidate in candidates:
                proposal = list(selected)
                proposal[weakest] = candidate
                q = self._score(student, proposal, train, proxy=True).require_objective("Proxy candidate evaluation")
                proposals.append((q, candidate, proposal))
            # Stable tie-break: first candidate in training order.
            proxy_q, candidate, proposal = max(proposals, key=lambda item: item[0])
            row = {
                "round": iteration, "selected_ids_before": [str(train[i].uid) for i in selected],
                "target_q_before": baseline_q, "marginal_contributions": contributions,
                "removed_id": str(train[selected[weakest]].uid), "candidate_id": str(train[candidate].uid),
                "proxy_candidate_q": proxy_q, "estimated_gain": proxy_q - baseline_q,
                "candidate_estimates": [{"uid": str(train[i].uid), "q": q} for q, i, _ in proposals],
                "accepted": False,
            }
            # The proxy ranks proposals. Only the target's gain decides whether
            # the best proposal qualifies; a pessimistic proxy is not a veto.
            proposed = self._score(student, proposal, train, proxy=False)
            proposed.require_objective("Proposed target evaluation")
            gain = proposed.objective - current.objective
            row.update(target_q_before=current.objective, target_q_proposed=proposed.objective, target_gain=gain)
            # Include the threshold itself; tolerate only floating-point round-off.
            if gain < self.epsilon and not math.isclose(gain, self.epsilon, rel_tol=1e-12, abs_tol=1e-12):
                row["stop_reason"] = self.stop_reason = "target_gain_below_epsilon"
                self.history.append(row)
                break
            selected, current = proposal, proposed
            row["accepted"] = True
            self.history.append(row)
        self.selected_indices = list(selected)
        self.final_metrics = current
        program = self._program(student, selected, train)
        program._compiled = True
        return program

    def report(self) -> dict:
        return {
            "algorithm": "correlation_guided_proxy_proposal_target_acceptance",
            "proxy_strategy": "exhaustive_LM_rescoring_of_training_set",
            "epsilon": self.epsilon, "max_iterations": self.max_iterations,
            "demo_count": self.demo_count, "seed": self.seed,
            "stop_reason": self.stop_reason,
            "target_training_set_evaluations": self.target_evaluations,
            "proxy_training_set_evaluations": self.proxy_evaluations,
            "history": self.history,
        }


# Backward-compatible optimizer import alias.
BootstrapFewShotWithCorrelationGuidedSearch = CorrelationGuidedOptimizer
