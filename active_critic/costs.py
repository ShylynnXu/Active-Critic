"""Measured client invocations, separate from protocol-derived call estimates."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class LMCallCounter:
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    _lock: Any = field(default_factory=Lock, repr=False)

    def __deepcopy__(self, memo):
        # DSPy copies scoring programs for demonstration subsets. Those copies
        # belong to the same target/proxy run and must share their actual counter.
        return self

    def start(self) -> None:
        with self._lock:
            self.attempted += 1

    def finish(self, success: bool) -> None:
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {key: getattr(self, key) for key in ("attempted", "completed", "failed")}


def call_snapshot(lm: Any) -> dict[str, int] | None:
    counter = getattr(lm, "call_counter", None)
    return None if counter is None else counter.snapshot()


def call_delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    return {key: after[key] - before[key] for key in ("attempted", "completed", "failed")}


def observed_costs(
    *, stage1_calls: int, training_instances: int, test_instances: int,
    optimization_target: dict | None, optimization_proxy: dict | None, test_target: dict | None,
) -> dict:
    target = None if optimization_target is None else stage1_calls + optimization_target["attempted"]
    proxy = None if optimization_proxy is None else optimization_proxy["attempted"]
    total = None if target is None or proxy is None else target + proxy
    return {
        "unit": "model_client_invocations_excluding_transport_retries",
        "stage1_target_calls": stage1_calls,
        "optimization_target": optimization_target,
        "optimization_proxy": optimization_proxy,
        "test_target": test_target,
        "construction_target_calls_per_instance": None if target is None else target / training_instances,
        "paper_cost_scope": "target_model_only",
        "paper_comparable_training_calls_per_instance": None if target is None else target / training_instances,
        "construction_proxy_calls_per_instance": None if proxy is None else proxy / training_instances,
        "construction_total_calls_per_instance": None if total is None else total / training_instances,
        "test_calls_per_instance": None if test_target is None else test_target["attempted"] / test_instances,
        "includes_adapter_level_extra_calls": True,
    }

