from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import DATASET_CONFIGS, load_dataset, preference_examples, split_examples
from .llm import MODEL_NAME, ModelConfig


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--dataset-dir", type=Path, default=root / "dataset")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    # Independent local proxy for demonstration selection.
    parser.add_argument("--proxy-model", default=os.getenv("ORCA2_MODEL_NAME", "orca2-7b"),
                        help="Locally served proxy model (default: orca2-7b)")
    parser.add_argument("--proxy-api-base", default=os.getenv("ORCA2_API_BASE", "http://127.0.0.1:8000/v1"),
                        help="Local OpenAI-compatible proxy endpoint; set its actual port")
    parser.add_argument("--proxy-api-key-env", default="ORCA2_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Audit data/configuration without model calls")


def model_config(args: argparse.Namespace, *, proxy: bool = False) -> ModelConfig:
    return ModelConfig(
        model=args.proxy_model if proxy else args.model,
        api_base=args.proxy_api_base if proxy else args.api_base,
        api_key_env=args.proxy_api_key_env if proxy else args.api_key_env,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )


def prepare_split(args: argparse.Namespace):
    examples = load_dataset(args.dataset, args.dataset_dir)
    supervision = getattr(args, "preference_supervision", None)
    if supervision:
        if args.dataset != "summeval":
            raise ValueError("OO/FO is defined for SummEval only")
        examples = preference_examples(examples, supervision)
    training, test = split_examples(examples, train_ratio=0.25, seed=args.split_seed)
    return examples, training, test


def new_run_directory(output: Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output / name / f"{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
