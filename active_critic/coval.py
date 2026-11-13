"""CoVal within-prompt criterion recovery; selection metadata never enters prompts."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Any, Sequence

from .schemas import Example
from .stage1 import Stage1Config, Stage1Inferer

LETTERS = ("A", "B", "C", "D")


def borda_scores(order: str) -> dict[str, float]:
    tiers = [[x.strip() for x in tier.split("=")] for tier in order.split(">")]
    flat = [x for tier in tiers for x in tier]
    if len(flat) != 4 or set(flat) != set(LETTERS):
        raise ValueError("Ranking must contain A, B, C, D exactly once")
    scores, remaining = {}, 4
    for tier in tiers:
        remaining -= len(tier)
        for label in tier:
            scores[label] = remaining + (len(tier) - 1) / 2
    return scores


def _utilities(value: Any) -> dict[str, float]:
    if isinstance(value, str):
        return borda_scores(value)
    if not isinstance(value, dict) or set(value) != set(LETTERS):
        raise ValueError("Expected a four-response ranking or Borda utility mapping")
    result = {k: float(v) for k, v in value.items()}
    if not all(math.isfinite(x) and 0 <= x <= 3 for x in result.values()) or not math.isclose(sum(result.values()), 6):
        raise ValueError("Invalid Borda scores (range 0-3; ties use average positions)")
    # Check that numeric ties correspond to an actual complete weak order.
    ordered = sorted(set(result.values()), reverse=True)
    order = ">".join("=".join(k for k in LETTERS if result[k] == value) for value in ordered)
    if borda_scores(order) != result:
        raise ValueError("Utilities do not correspond to an average-rank Borda order")
    return result


def prepare_groups(selected_rows: Sequence[dict], *, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Read an author-selected one-criterion-per-prompt manifest/expA_candidates.

    Do not choose a criterion by the observed ranking difference. The existing
    expA_candidates export already freezes that independent selection.
    """
    if not selected_rows:
        raise ValueError("The frozen CoVal selection must contain at least one prompt")
    public, gold = [], []
    seen = set()
    rng = random.Random(seed)
    for index, row in enumerate(selected_rows):
        if row["prompt_id"] in seen:
            raise ValueError("Select exactly one criterion per prompt")
        seen.add(row["prompt_id"])
        if set(row["responses"]) != set(LETTERS):
            raise ValueError("Every prompt must have the same four candidate responses")
        group_names = ["group_support", "group_avoid"]
        rng.shuffle(group_names)
        members = [set(row[g]["annotators"]) for g in group_names]
        if members[0] & members[1]:
            raise ValueError("Support and Avoid groups must be disjoint")
        for position, name in enumerate(group_names):
            group = row[name]
            users = group["annotators"]
            weights = group["weights"]
            rankings = group["personal_rankings"]
            if len(users) < 3 or len(users) != len(set(users)) or len(weights) != len(users) or set(rankings) != set(users):
                raise ValueError("Each group needs at least three distinct ranked and weighted participants")
            if not all(math.isfinite(float(w)) and -10 <= w <= 10 for w in weights):
                raise ValueError("CoVal weights must lie in [-10,10]")
            if not all(w >= 5 if name == "group_support" else w <= -5 for w in weights):
                raise ValueError("Support/Avoid weight threshold is not satisfied")
            utilities = [_utilities(rankings[u]) for u in users]
            means = {k: sum(x[k] for x in utilities) / len(utilities) for k in LETTERS}
            uid = f"case_{index:04d}_{position}"
            # Whitelist only model-allowed fields. No user IDs, criterion, weights,
            # support/avoid names, rankings of the other group, or diagnostics.
            public.append({
                "uid": uid, "case_id": f"case_{index:04d}",
                "source": json.dumps([
                    {"role": m["role"], "content": m["content"]} for m in row["prompt_messages"]
                ], ensure_ascii=False),
                "responses": [
                    {"candidate": k, "response": row["responses"][k], "human_score": means[k]}
                    for k in LETTERS
                ],
            })
            gold.append({
                "uid": uid, "case_id": f"case_{index:04d}", "criterion": row["criterion"],
                "direction": "support" if name == "group_support" else "avoid",
                "source_prompt_id": row["prompt_id"], "participants": len(users),
            })
    return public, gold


class _BordaLM:
    def __init__(self, llm: Any):
        self.llm = llm

    def generate(self, prompt: str) -> str:
        return self.llm.generate(
            "In this experiment, Human Score, when present, is the group-mean Borda utility on a 0-3 scale, "
            "with larger values indicating stronger preference. It is not an absolute quality rating. "
            "Infer criterion content and preferred direction from the supplied text and any available feedback. "
            "If feedback is absent, do not invent group preferences.\n\n" + prompt
        )


def _validate_model_inputs(records: Sequence[dict], *, no_feedback: bool) -> list[dict]:
    """Check the whole input before calling the model and fix candidate order."""
    validated, seen, text_by_case = [], set(), {}
    if not records:
        raise ValueError("CoVal inference requires at least one group")
    for record in records:
        uid, key, source = record.get("uid"), record.get("case_id"), record.get("source")
        if any(not isinstance(x, str) or not x.strip() for x in (uid, key, source)):
            raise ValueError("Each CoVal group needs nonempty uid, case_id and source strings")
        if uid in seen:
            raise ValueError("CoVal group UIDs must be unique")
        seen.add(uid)
        items = record.get("responses")
        if not isinstance(items, list) or len(items) != 4 or any(not isinstance(x, dict) for x in items):
            raise ValueError("Each CoVal group needs four candidate responses")
        labels = [x.get("candidate") for x in items]
        if sorted(str(x) for x in labels) != list(LETTERS):
            raise ValueError("Candidate IDs must be A, B, C, D exactly once")
        by_candidate = {x["candidate"]: x for x in items}
        responses = []
        for letter in LETTERS:
            item = by_candidate[letter]
            response = item.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError("CoVal response text must be a nonempty string")
            public = {"candidate": letter, "response": response}
            if not no_feedback:
                score = item.get("human_score")
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 3):
                    raise ValueError("Group-mean Borda scores must be finite numbers in [0, 3]")
                public["human_score"] = float(score)
            responses.append(public)
        if not no_feedback and not math.isclose(sum(x["human_score"] for x in responses), 6):
            raise ValueError("The four group-mean Borda utilities must sum to 6")
        fingerprint = (source, tuple(x["response"] for x in responses))
        if key in text_by_case and text_by_case[key] != fingerprint:
            raise ValueError("The two groups of a prompt must have identical source and responses")
        text_by_case[key] = fingerprint
        validated.append({"uid": uid, "case_id": key, "source": source, "responses": responses})
    return validated


def infer_groups(records: Sequence[dict], llm: Any, *, seed: int = 42, no_feedback: bool = False) -> list[dict]:
    """The control changes feedback fields only, retaining the same Stage-1 pipeline."""
    records = _validate_model_inputs(records, no_feedback=no_feedback)
    results, controls = [], {}
    for record in records:
        key = record["case_id"]
        if no_feedback and key in controls:
            result = controls[key]
        else:
            examples = [
                Example(uid=x["candidate"], source=record["source"], response=x["response"],
                        human_score=None if no_feedback else float(x["human_score"]),
                        raw_score=None if no_feedback else float(x["human_score"]))
                for x in record["responses"]
            ]
            inferer = Stage1Inferer(
                _BordaLM(llm), Stage1Config(seed=seed, use_feedback=not no_feedback),
            )
            instruction = inferer.run(examples)
            result = {
                "criteria": instruction.criteria, "task_description": instruction.task_description,
                "trace": inferer.trace,
            }
            if no_feedback:
                controls[key] = result
        results.append({"uid": record["uid"], "case_id": key, "feedback_used": not no_feedback, **result})
    return results


def recovery_metrics(gold: Sequence[dict], judgments: Sequence[dict], *, no_feedback: bool = False) -> dict:
    """Semantic correctness must be independently adjudicated, not string-matched.

    Judgment row: uid, criterion_correct (bool), direction (support/avoid/undetermined).
    Gold is used only here, never by the inference program. Paired discrimination
    is not applicable for the shared-output control.
    """
    truth = {x["uid"]: x for x in gold}
    predicted = {x["uid"]: x for x in judgments}
    if len(truth) != len(gold) or len(predicted) != len(judgments) or set(truth) != set(predicted) or not truth:
        raise ValueError("Judgments must cover every gold instance exactly once")
    correct, opposite, pairs = 0, 0, defaultdict(list)
    criteria_by_case = {}
    for uid, item in truth.items():
        if item["direction"] not in {"support", "avoid"}:
            raise ValueError("Gold directions must be Support or Avoid")
        key, criterion = item["case_id"], item["criterion"]
        if key in criteria_by_case and criteria_by_case[key] != criterion:
            raise ValueError("Both groups must concern the same selected criterion")
        criteria_by_case[key] = criterion
        judgment = predicted[uid]
        if type(judgment["criterion_correct"]) is not bool or judgment["direction"] not in {"support", "avoid", "undetermined"}:
            raise ValueError("Invalid semantic adjudication")
        recovered = judgment["criterion_correct"] and judgment["direction"] == item["direction"]
        correct += recovered
        opposite += judgment["criterion_correct"] and judgment["direction"] in {"support", "avoid"} and not recovered
        pairs[item["case_id"]].append((item["direction"], recovered))
    if any(len(x) != 2 or {g for g, _ in x} != {"support", "avoid"} for x in pairs.values()):
        raise ValueError("Each prompt needs one Support and one Avoid judgment")
    return {
        "group_instances": len(gold), "prompts": len(pairs),
        "feedback_used": not no_feedback,
        "recovery_rate": correct / len(gold),
        "paired_discrimination_applicable": not no_feedback,
        "paired_discrimination_rate": (
            None if no_feedback else sum(all(ok for _, ok in x) for x in pairs.values()) / len(pairs)
        ),
        "opposite_direction_error": opposite / len(gold),
    }

