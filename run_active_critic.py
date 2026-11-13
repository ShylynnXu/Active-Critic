from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from active_critic.cli import add_common_arguments, model_config, new_run_directory, prepare_split
from active_critic.costs import call_delta, call_snapshot, observed_costs
from active_critic.data import DATASET_CONFIGS, audit_split, labeled_subset
from active_critic.dspy_program import configure_dspy_lm, create_dspy_lm
from active_critic.io import read_json, write_json
from active_critic.llm import OpenAILanguageModel
from active_critic.parsing import parse_criteria
from active_critic.pipeline import (
    DEMONSTRATION_COUNTS, evaluate_program, file_sha256, fit_program,
    infer_instruction, runtime_versions, save_evaluator, save_split,
)
from active_critic.stage1 import Stage1Config
from active_critic.statistics import summarize_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Active-Critic task inference and scoring alignment.")
    add_common_arguments(parser)
    parser.add_argument("--mode", required=True, choices=("coarse", "fine"))
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--seeds", nargs="+", type=int, help="Independent runs; split-seed stays fixed")
    runs.add_argument("--repeats", type=int, default=1, help="Use seed, seed+1, ...; e.g. 5")
    parser.add_argument("--budget", default="all", help="all, 5, 5%%, 15%%, or 25%%")
    parser.add_argument("--demo-count", type=int, help="Default 3/3/8/8, bounded by available labels")
    parser.add_argument("--no-opt", action="store_true")
    parser.add_argument("--single-batch", action="store_true")
    parser.add_argument("--no-task-description", action="store_true")
    parser.add_argument("--no-explanations", action="store_true")
    parser.add_argument("--aggregation", choices=("llm", "mean"), default="llm")
    parser.add_argument("--criteria-file", type=Path, help="Exact predefined criterion definitions, JSON")
    parser.add_argument("--preference-supervision", choices=("OO", "FO"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1 or args.bootstrap_resamples < 0:
        raise ValueError("Invalid repetitions or bootstrap count")
    if args.aggregation == "mean" and args.mode != "fine":
        raise ValueError("--aggregation mean requires --mode fine")
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.repeats))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated runs must have distinct seeds")
    override = parse_criteria(read_json(args.criteria_file), 1, None) if args.criteria_file else None
    examples, training_pool, test = prepare_split(args)
    config = model_config(args)
    proxy_config = None if args.no_opt else model_config(args, proxy=True)
    prepared = []
    for seed in seeds:
        train = labeled_subset(training_pool, args.budget, len(examples), seed)
        count = args.demo_count if args.demo_count is not None else min(DEMONSTRATION_COUNTS[args.dataset], len(train))
        if not 1 <= count <= len(train):
            raise ValueError("Demonstrations cannot exceed the available label budget")
        prepared.append((seed, train, count))
    summary = {
        "dataset": args.dataset, "mode": args.mode, "seeds": seeds,
        "split_seed": args.split_seed, "split": audit_split(training_pool, test),
        "target": config.to_dict(), "proxy": None if args.no_opt else proxy_config.to_dict(),
        "runs": [{"seed": s, "training_labels": len(t), "demonstrations": n} for s, t, n in prepared],
        "budget": args.budget, "stage1_max_batches": 1 if args.single_batch else 25,
        "consecutive_complete": 5, "epsilon": 0.01, "max_refinement_rounds": 11,
        "preference_supervision": args.preference_supervision,
        "optimization_enabled": not args.no_opt,
        "task_description_inference": not args.no_task_description,
        "criteria_source": "provided" if override is not None else "inferred",
        "explanations": not args.no_explanations, "aggregation": args.aggregation,
        "bootstrap_resamples": args.bootstrap_resamples,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    target = configure_dspy_lm(config)
    proxy = None if args.no_opt else create_dspy_lm(proxy_config)
    directory = new_run_directory(args.output_dir, f"{args.dataset}_{args.mode}")
    summary["dataset_sha256"] = file_sha256(args.dataset_dir / DATASET_CONFIGS[args.dataset].filename)
    summary["runtime_versions"] = runtime_versions()
    root = Path(__file__).resolve().parent
    summary["source_sha256"] = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py")) if "output" not in p.parts and "__pycache__" not in p.parts}
    if args.criteria_file:
        summary["criteria_file_sha256"] = file_sha256(args.criteria_file)
    write_json(directory / "manifest.json", summary)
    save_split(directory, training_pool, test, args.split_seed)
    metrics = []
    for seed, train, count in prepared:
        run_dir = directory / f"seed_{seed}"
        run_dir.mkdir()
        save_split(run_dir, train, test, args.split_seed)
        stage1_config = Stage1Config(
            seed=seed, max_batches=1 if args.single_batch else 25,
            infer_task=not args.no_task_description,
            shuffle_examples=False,
        )
        # Always repeat task inference from scratch: there is no implicit disk cache.
        stage1_lm = OpenAILanguageModel(config=config)
        instruction = infer_instruction(train, stage1_lm, stage1_config, run_dir, criteria_override=override)
        target_before_fit, proxy_before_fit = call_snapshot(target), call_snapshot(proxy)
        program, report = fit_program(
            instruction, train, mode=args.mode, target_lm=target, proxy_lm=proxy,
            demo_count=count, seed=seed, optimize=not args.no_opt,
            explanations=not args.no_explanations, aggregation=args.aggregation,
        )
        target_after_fit, proxy_after_fit = call_snapshot(target), call_snapshot(proxy)
        write_json(run_dir / "optimization.json", report)
        save_evaluator(program, run_dir, config.to_dict())
        metrics.append(evaluate_program(program, test, run_dir, seed=seed, bootstrap_resamples=args.bootstrap_resamples))
        measured_calls = observed_costs(
            stage1_calls=stage1_lm.calls, training_instances=len(train), test_instances=len(test),
            optimization_target=call_delta(target_before_fit, target_after_fit),
            optimization_proxy=({"attempted": 0, "completed": 0, "failed": 0} if args.no_opt
                                else call_delta(proxy_before_fit, proxy_after_fit)),
            test_target=call_delta(target_after_fit, call_snapshot(target)),
        )
        module_calls = len(program.predictors())
        optimization_target_calls = report.get("target_training_set_evaluations", 0) * len(train) * module_calls
        optimization_proxy_calls = report.get("proxy_training_set_evaluations", 0) * len(train) * module_calls
        write_json(run_dir / "cost.json", {
            "unit": "logical_module_calls_excluding_transport_and_format_retries",
            "stage1_target_calls": stage1_lm.calls,
            "optimization_target_calls": optimization_target_calls,
            "optimization_proxy_calls": optimization_proxy_calls,
            "training_instances": len(train),
            "construction_target_calls_per_instance": (stage1_lm.calls + optimization_target_calls) / len(train),
            "construction_proxy_calls_per_instance": optimization_proxy_calls / len(train),
            "construction_total_calls_per_instance": (stage1_lm.calls + optimization_target_calls + optimization_proxy_calls) / len(train),
            "test_calls": len(test) * module_calls,
            "test_calls_per_instance": module_calls,
            "stage1_input_tokens": stage1_lm.input_tokens,
            "stage1_output_tokens": stage1_lm.output_tokens,
            "observed_lm_calls": measured_calls,
        })
    stability = summarize_runs(metrics)
    write_json(directory / "summary.json", stability)
    print(json.dumps({"output": str(directory), **stability}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
