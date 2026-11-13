from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Iterable

from .llm import LanguageModel
from .parsing import OutputFormatError, get_case_insensitive, parse_criteria, parse_json_object
from .prompts import (
    CRITERIA_OBSERVATION_PROMPT, INITIAL_CRITERIA_PROMPT, REFINE_CRITERIA_PROMPT,
    TASK_DESCRIPTION_PROMPT, TASK_ENSEMBLE_PROMPT, TASK_OBSERVATION_PROMPT,
)
from .schemas import Example, TaskInstruction


@dataclass(frozen=True)
class Stage1Config:
    batch_size: int = 5
    max_batches: int = 25
    consecutive_complete: int = 5
    min_criteria: int = 4
    max_criteria: int = 10
    seed: int = 42
    infer_task: bool = True
    shuffle_examples: bool = True
    use_feedback: bool = True

    def __post_init__(self) -> None:
        if min(self.batch_size, self.max_batches, self.consecutive_complete) <= 0:
            raise ValueError("Batch sizes and stopping limits must be positive")
        if not 1 <= self.min_criteria <= self.max_criteria:
            raise ValueError("Invalid criterion-count bounds")


def make_minibatches(examples: Iterable[Example], config: Stage1Config) -> list[list[Example]]:
    """Visit each label at most once; do not duplicate small labeled budgets."""
    items = list(examples)
    if not items:
        raise ValueError("Stage 1 requires training examples")
    if config.shuffle_examples:
        random.Random(config.seed).shuffle(items)
    return [
        items[start:start + config.batch_size]
        for start in range(0, min(len(items), config.max_batches * config.batch_size), config.batch_size)
    ]


def _examples_json(batch: list[Example], *, include_scores: bool = True) -> str:
    return json.dumps([item.prompt_record(include_score=include_scores) for item in batch], ensure_ascii=False, indent=2)


def _is_complete(text: str) -> bool:
    return bool(re.fullmatch(r"""\s*["'\x60‘’“”]*COMPLETE["'\x60‘’“”]*[.!]?\s*""", text, re.I))


def _parse_task_description(text: str) -> tuple[str, str, str]:
    description = " ".join(text.strip().split())
    task = re.search(r"the\s+evaluation\s+task\s+is\s+(.+?),\s*which\s+refers\s+to\s+evaluating\b", description, re.I)
    relationship = re.search(
        r"""the\s+relationship\s+between\s+[‘’“”"']?Source[‘’“”"']?\s+and\s+[‘’“”"']?Response[‘’“”"']?\s+is\s+(.+)$""",
        description, re.I,
    )
    if not task or not relationship:
        raise OutputFormatError("Task description did not follow the requested summary format")
    return task.group(1).strip(), description, relationship.group(1).strip()


class Stage1Inferer:
    """Iterative task and criterion inference with consecutive-completion stopping."""

    def __init__(self, llm: LanguageModel, config: Stage1Config | None = None) -> None:
        self.llm = llm
        self.config = config or Stage1Config()
        self.trace: list[dict] = []
        self.task_descriptions: list[str] = []

    def _call(self, phase: str, prompt: str, batch: list[Example] | None = None) -> str:
        response = self.llm.generate(prompt).strip()
        self.trace.append({
            "phase": phase, "batch_ids": [x.uid for x in batch or []],
            "prompt": prompt, "response": response,
        })
        return response

    def infer_task_description(self, batches: list[list[Example]]) -> tuple[str, str, str]:
        observations: list[str] = []
        self.task_descriptions = []
        consecutive = 0
        for batch in batches[:self.config.max_batches]:
            observation = self._call("task_observation", TASK_OBSERVATION_PROMPT.format(
                examples=_examples_json(batch, include_scores=self.config.use_feedback),
                prior_observations=json.dumps(observations, ensure_ascii=False),
            ), batch)
            if _is_complete(observation):
                if not self.task_descriptions:
                    raise OutputFormatError("COMPLETE cannot precede the first task description")
                consecutive += 1
                if consecutive == self.config.consecutive_complete:
                    break
                continue
            if not observation:
                raise OutputFormatError("Empty task observation")
            consecutive = 0
            observations.append(observation)
            description = self._call("task_description", TASK_DESCRIPTION_PROMPT.format(
                observations=json.dumps([observation], ensure_ascii=False),
                prior_task_description="",
            ), batch)
            self.task_descriptions.append(_parse_task_description(description)[1])
        if not self.task_descriptions:
            raise OutputFormatError("No task descriptions were produced")
        ensemble = self._call("task_ensemble", TASK_ENSEMBLE_PROMPT.format(
            task_descriptions=json.dumps(self.task_descriptions, ensure_ascii=False, indent=2),
        ))
        return _parse_task_description(ensemble)

    def infer_criteria(
        self, task_type: str, batches: list[list[Example]], task_description: str = "",
    ) -> dict[str, str]:
        if not batches:
            raise ValueError("Criteria inference requires a mini-batch")
        initial = parse_json_object(self._call("criteria_initial", INITIAL_CRITERIA_PROMPT.format(
            task_type=task_type, task_description=task_description,
            examples=_examples_json(batches[0], include_scores=self.config.use_feedback),
        ), batches[0]))
        # The initial template has no numerical cardinality limit. The 4-10
        # instruction belongs to refinement, which small budgets may never reach.
        criteria = parse_criteria(initial, 1, None)
        observation = get_case_insensitive(initial, "Observations")
        if not isinstance(observation, str) or not observation.strip():
            raise OutputFormatError("Initial criteria must include observations")
        observations = [observation.strip()]
        consecutive = 0
        # Initial generation is batch 1; at most 25 batches in total.
        for batch in batches[1:self.config.max_batches]:
            observation = self._call("criteria_observation", CRITERIA_OBSERVATION_PROMPT.format(
                task_type=task_type, task_description=task_description,
                examples=_examples_json(batch, include_scores=self.config.use_feedback),
                prior_observations=json.dumps(observations, ensure_ascii=False),
                current_evaluation_criteria=json.dumps(criteria, ensure_ascii=False),
            ), batch)
            if _is_complete(observation):
                consecutive += 1
                if consecutive == self.config.consecutive_complete:
                    break
                continue
            if not observation:
                raise OutputFormatError("Empty criteria observation")
            consecutive = 0
            observations.append(observation)
            criteria = parse_criteria(parse_json_object(self._call(
                "criteria_refine", REFINE_CRITERIA_PROMPT.format(
                    task_description=task_description,
                    min_criteria=self.config.min_criteria, max_criteria=self.config.max_criteria,
                    evaluation_criteria=json.dumps(criteria, ensure_ascii=False),
                    new_observations=observation,
                ), batch,
            )), self.config.min_criteria, self.config.max_criteria)
        return criteria

    def run(
        self, training_examples: Iterable[Example], *, criteria_override: dict[str, str] | None = None,
    ) -> TaskInstruction:
        self.trace = []
        batches = make_minibatches(training_examples, self.config)
        if self.config.infer_task:
            task_type, description, relationship = self.infer_task_description(batches)
        else:
            task_type, description, relationship = "natural language generation", "", ""
        criteria = (
            parse_criteria(criteria_override, 1, None) if criteria_override is not None
            else self.infer_criteria(task_type, batches, description)
        )
        return TaskInstruction(task_type, description, relationship, criteria)
