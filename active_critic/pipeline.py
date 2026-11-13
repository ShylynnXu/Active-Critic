from __future__ import annotations

import hashlib
import importlib.metadata
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .data import audit_split
from .dspy_optimizer import CorrelationGuidedOptimizer
from .dspy_program import ActiveCriticProgram, DSPyActiveCriticScorer, make_dspy_examples, save_program_state
from .io import write_json, write_jsonl
from .metrics import calculate_correlations
from .schemas import Example, TaskInstruction
from .stage1 import Stage1Config, Stage1Inferer
from .statistics import pearson_bootstrap_ci

DEMONSTRATION_COUNTS = {"summeval": 3, "topicalchat": 3, "sfres": 8, "openmeva": 8}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("dspy", "openai", "scipy", "numpy")}


def infer_instruction(
    training: Sequence[Example], llm: Any, config: Stage1Config, directory: Path,
    *, criteria_override: dict[str, str] | None = None,
) -> TaskInstruction:
    inferer = Stage1Inferer(llm, config)
    instruction = inferer.run(training, criteria_override=criteria_override)
    write_json(directory / "stage1_instruction.json", instruction.to_dict())
    write_jsonl(directory / "stage1_trace.jsonl", inferer.trace)
    write_json(directory / "stage1_config.json", asdict(config))
    return instruction


def fit_program(
    instruction: TaskInstruction, training: Sequence[Example], *, mode: str, target_lm: Any,
    proxy_lm: Any, demo_count: int, seed: int, optimize: bool = True,
    explanations: bool = True, aggregation: str = "llm",
) -> tuple[Any, dict]:
    examples = make_dspy_examples(training)
    if not 1 <= demo_count <= len(examples):
        raise ValueError("Demo count must fit within the available labeled budget")
    program = ActiveCriticProgram(instruction, mode, explanations=explanations, aggregation=aggregation)
    program.set_lm(target_lm)
    initial = random.Random(seed).sample(range(len(examples)), demo_count)
    if optimize:
        optimizer = CorrelationGuidedOptimizer(demo_count=demo_count, proxy_lm=proxy_lm, seed=seed)
        program = optimizer.compile(program, trainset=examples, initial_indices=initial)
        report = optimizer.report()
        selected = optimizer.selected_indices
    else:
        program.set_demonstrations([examples[i] for i in initial])
        report = {"algorithm": "fixed_random_demonstrations", "seed": seed, "history": []}
        selected = initial
    report["initial_demo_ids"] = [training[i].uid for i in initial]
    report["selected_demo_ids"] = [training[i].uid for i in selected]
    report["selected_demonstrations"] = [
        {"uid": training[i].uid, **training[i].prompt_record()} for i in selected
    ]
    return program, report


def evaluate_program(program: Any, test: Sequence[Example], directory: Path, *, seed: int, bootstrap_resamples: int = 1000) -> dict:
    scorer = DSPyActiveCriticScorer(program, program.instruction, program.mode)
    rows = [scorer.score(x).to_dict() for x in test]
    return save_prediction_results(rows, directory, seed=seed, bootstrap_resamples=bootstrap_resamples)


def save_prediction_results(rows: Sequence[dict], directory: Path, *, seed: int, bootstrap_resamples: int = 1000) -> dict:
    """Apply the same statistical procedure to full and intervention outputs."""
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be nonnegative")
    predictions, labels = [x["overall"]["score"] for x in rows], [x["human_score"] for x in rows]
    metrics = calculate_correlations(predictions, labels).to_dict()
    write_jsonl(directory / "predictions.jsonl", rows)
    if bootstrap_resamples:
        metrics["pearson_95ci"] = pearson_bootstrap_ci(
            predictions, labels, seed=seed, resamples=bootstrap_resamples,
        )
    write_json(directory / "metrics.json", metrics)
    return metrics


def save_evaluator(program: Any, directory: Path, model_config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_program_state(program, directory / "dspy_program.json")
    write_json(directory / "evaluator.json", {
        "instruction": program.instruction.to_dict(), "mode": program.mode,
        "explanations": program.explanations, "aggregation": program.aggregation,
        "model": model_config, "demonstration_ids": program.demonstration_ids,
    })


def load_evaluator(directory: Path, lm: Any) -> Any:
    from .io import read_json

    config = read_json(directory / "evaluator.json")
    program = ActiveCriticProgram(
        TaskInstruction.from_dict(config["instruction"]), config["mode"],
        explanations=config["explanations"], aggregation=config["aggregation"],
    )
    program.load(str(directory / "dspy_program.json"))
    program.demonstration_ids = config["demonstration_ids"]
    program.set_lm(lm)
    return program


def save_split(directory: Path, train: Sequence[Example], test: Sequence[Example], seed: int) -> None:
    write_json(directory / "split.json", {
        "split_seed": seed, "audit": audit_split(train, test),
        "train_ids": [x.uid for x in train], "test_ids": [x.uid for x in test],
    })
