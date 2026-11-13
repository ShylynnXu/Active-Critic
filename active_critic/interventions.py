"""Controlled OS-E input masking and shuffling with fixed scoring programs."""
from __future__ import annotations

import random
import re
from typing import Any, Sequence

from .schemas import CriterionResult, Example

_SMALL_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_WORD_VALUES = {word: i for i, word in enumerate(_SMALL_WORDS)}
_WORD_VALUES.update({word: 20 + 10 * i for i, word in enumerate(_TENS)})
_ENGLISH_INTEGER = (
    r"(?:one[\s-]+hundred|hundred|(?:" + "|".join(_TENS) + r")(?:[\s-]+(?:"
    + "|".join(_SMALL_WORDS[1:10]) + r"))?|" + "|".join(reversed(_SMALL_WORDS)) + r")"
)
_ENGLISH_NUMBER = _ENGLISH_INTEGER + r"(?:\s+point\s+(?:" + "|".join(_SMALL_WORDS[:10]) + r")(?:\s+(?:" + "|".join(_SMALL_WORDS[:10]) + r"))*)?"
_NUMBER_TEXT = r"(?<![\w.])(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|" + _ENGLISH_NUMBER + r")(?!\w|\.\d)"
_NUMBER = re.compile(_NUMBER_TEXT, re.I)
_RATIO = re.compile(r"(?P<a>" + _NUMBER_TEXT + r")\s*(?:/|out\s+of)\s*(?P<b>" + _NUMBER_TEXT + r")", re.I)
_SCORE_CONTEXT = re.compile(
    r"(?P<prefix>\b(?:scores?|ratings?|rated|grades?|graded|rate|scored)\b"
    r"(?:\s+(?:is|was|of|at|as|it|this|the|response|a|an|overall|only|about|approximately|around|would|be))*\s*(?:[:=]\s*)?)"
    r"(?P<number>" + _NUMBER_TEXT + r")", re.I,
)


def _number_value(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        words = text.lower().replace("-", " ").split()
        if "point" in words:
            split = words.index("point")
            integer = _number_value(" ".join(words[:split]))
            decimal = "".join(str(_WORD_VALUES[x]) for x in words[split + 1:])
            return integer + float("0." + decimal)
        if "hundred" in words:
            return 100.0
        return float(sum(_WORD_VALUES[x] for x in words))


def mask_score_mentions(explanation: str, score: float) -> str:
    """Hide score values, preserving unrelated dates, counts, and other evidence.

    Redact explicit score/rating expressions, equivalent score fractions, and
    numeric or English-word occurrences of the actual criterion score. A bare
    value equal to the score is masked conservatively because its role is
    ambiguous; unrelated numbers are not removed.
    """
    import math

    def ratio(match: re.Match) -> str:
        numerator, denominator = _number_value(match["a"]), _number_value(match["b"])
        if denominator > 0 and math.isclose(100 * numerator / denominator, score, abs_tol=1e-8):
            return "[MASKED]"
        return match.group(0)

    text = _RATIO.sub(ratio, explanation)
    text = _SCORE_CONTEXT.sub(lambda match: match["prefix"] + "[MASKED]", text)
    return _NUMBER.sub(
        lambda match: "[MASKED]" if math.isclose(_number_value(match.group(0)), score, abs_tol=1e-8)
        else match.group(0), text,
    )


def mask_assessments(assessments: dict[str, CriterionResult]) -> dict:
    return {
        name: {"score": "[MASKED]", "explanation": mask_score_mentions(item.explanation, item.score)}
        for name, item in assessments.items()
    }


def shuffled_donors(size: int, seed: int) -> list[int]:
    """Ordinary seeded sample permutation; fixed points are permitted."""
    if size < 2:
        raise ValueError("Shuffle intervention needs at least two samples")
    permutation = list(range(size))
    random.Random(seed).shuffle(permutation)
    return permutation


def evaluate_interventions(program: Any, examples: Sequence[Example], seed: int) -> dict:
    if program.mode != "fine" or program.aggregation != "llm" or not program.explanations:
        raise ValueError("Interventions require the full AC-Fine with explanations")
    examples = list(examples)
    donors = shuffled_donors(len(examples), seed)
    # Compute each McS-E result exactly once, then keep it fixed in all comparisons.
    intermediates = [program.assess_criteria(source=x.source, response=x.response) for x in examples]
    results = {key: [] for key in ("complete", "mask", "shuffle")}
    for condition in results:
        for i, example in enumerate(examples):
            donor = donors[i] if condition == "shuffle" else i
            assessments = intermediates[donor]
            evidence = mask_assessments(assessments) if condition == "mask" else {
                name: item.to_dict() for name, item in assessments.items()
            }
            explanation, score = program.score_overall(
                source=example.source, response=example.response, evidence=evidence,
            )
            results[condition].append({
                "uid": example.uid, "human_score": example.human_score,
                "source": example.source, "response": example.response,
                "intermediate_donor_uid": examples[donor].uid,
                "os_e_input": evidence,
                "overall": {"explanation": explanation, "score": score},
            })
    return results

