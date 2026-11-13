from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic_core import core_schema

from .costs import LMCallCounter
from .llm import ModelConfig, prepare_endpoint
from .parsing import OutputFormatError, clamp_score, get_case_insensitive, parse_json_object
from .prompts import (
    COARSE_SCORING_INSTRUCTIONS, COARSE_SCORE_ONLY_INSTRUCTIONS, MULTI_CRITERIA_SCORING_INSTRUCTIONS,
    MULTI_CRITERIA_SCORE_ONLY_INSTRUCTIONS, OVERALL_SCORING_INSTRUCTIONS,
    OVERALL_SCORE_ONLY_INSTRUCTIONS, stage2_instructions,
)
from .schemas import CriterionResult, EvaluationResult, Example, TaskInstruction

Mode = Literal["coarse", "fine"]
try:
    import dspy as _dspy
except ImportError:
    _dspy = None


def require_dspy() -> Any:
    if _dspy is None:
        raise RuntimeError("DSPy is required. Install requirements.txt.")
    return _dspy


def create_dspy_lm(config: ModelConfig) -> Any:
    require_dspy()
    prepare_endpoint(config.api_base)
    return CountedLM(
        f"openai/{config.model}", model_type="chat", api_key=config.api_key(),
        api_base=config.api_base, temperature=config.temperature,
        max_tokens=config.max_tokens, cache=False, num_retries=2,
    )


def configure_dspy_lm(config: ModelConfig | None = None) -> Any:
    dspy = require_dspy()
    lm = create_dspy_lm(config or ModelConfig())
    dspy.configure(lm=lm, adapter=ActiveCriticJSONAdapter(use_native_function_calling=False))
    return lm


def _normalized_key(value: str) -> str:
    return "".join(c for c in value.casefold() if c.isalnum())


def parse_criterion_assessments(
    value: str | dict, criteria: dict[str, str], *, explanations: bool = True,
) -> dict[str, CriterionResult]:
    payload = parse_json_object(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise OutputFormatError("Criterion assessments must be a JSON object")
    wrapped = get_case_insensitive(payload, "Score Of Each Criterion In JSON")
    if isinstance(wrapped, dict):
        if len(payload) != 1:
            raise OutputFormatError("Unexpected fields outside the criterion assessments")
        payload = wrapped
    expected = {_normalized_key(name): name for name in criteria}
    if len(expected) != len(criteria):
        raise OutputFormatError("Criterion names are indistinguishable after normalization")
    received = {_normalized_key(str(name)): item for name, item in payload.items()}
    if len(received) != len(payload) or set(received) != set(expected):
        raise OutputFormatError("Returned criterion keys must exactly match the supplied criteria")
    results = {}
    for key, name in expected.items():
        item = received[key]
        if not isinstance(item, dict):
            raise OutputFormatError(f"Criterion {name!r} is not an object")
        fields = {str(field).casefold() for field in item}
        expected_fields = {"score", "explanation"} if explanations else {"score"}
        if len(fields) != len(item) or fields != expected_fields:
            raise OutputFormatError(
                f"Criterion {name!r} must contain exactly {sorted(expected_fields)}"
            )
        explanation = get_case_insensitive(item, "explanation", "")
        if not isinstance(explanation, str) or (explanations and not explanation.strip()):
            raise OutputFormatError(f"Criterion {name!r} omitted its explanation")
        results[name] = CriterionResult(
            explanation=explanation.strip(), score=clamp_score(get_case_insensitive(item, "score")),
        )
    return results


class EvaluationScore(float):
    """Keep DSPy's type conversion from accepting booleans as numeric scores."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.float_schema(strict=True, ge=0, le=100, allow_inf_nan=False)


class EvaluationExplanation(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.str_schema(strict=True, strip_whitespace=True, min_length=1)


if _dspy is not None:
    class CountedLM(_dspy.LM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_counter = LMCallCounter()

        def __call__(self, *args, **kwargs):
            self.call_counter.start()
            success = False
            try:
                result = super().__call__(*args, **kwargs)
                success = True
                return result
            finally:
                self.call_counter.finish(success)


    class ActiveCriticJSONAdapter(_dspy.JSONAdapter):
        def parse(self, signature, completion):
            # Validate before DSPy's JSON repair could discard duplicate fields.
            from dspy.utils.exceptions import AdapterParseError
            try:
                payload = parse_json_object(completion)
                if set(payload) != set(signature.output_fields):
                    raise OutputFormatError(
                        f"Expected exactly {sorted(signature.output_fields)}, received {sorted(payload)}. "
                        "Unexpected outputs, including explanations in score-only ablations, cannot be discarded."
                    )
                field = "score_of_each_criterion_in_json"
                if isinstance(payload.get(field), str):
                    # DSPy's dict conversion uses JSON repair on strings. Parse
                    # this inner payload first so duplicates/prose cannot vanish.
                    payload[field] = parse_json_object(payload[field])
            except OutputFormatError as exc:
                raise AdapterParseError(
                    adapter_name=type(self).__name__, signature=signature,
                    lm_response=completion, message=str(exc),
                ) from exc
            return super().parse(signature, json.dumps(payload, ensure_ascii=False))


    class MultiCriteriaSignature(_dspy.Signature):
        source: str = _dspy.InputField()
        response: str = _dspy.InputField()
        multiple_evaluation_criteria: str = _dspy.InputField(desc="Criterion names and definitions, in JSON")
        score_of_each_criterion_in_json: dict[str, dict[str, Any]] = _dspy.OutputField(desc="One JSON entry per criterion")


    class OverallSignature(_dspy.Signature):
        source: str = _dspy.InputField()
        response: str = _dspy.InputField()
        criterion_information: str = _dspy.InputField(desc="Criterion evidence or definitions in JSON")
        explanation: EvaluationExplanation = _dspy.OutputField()
        score_of_overall: EvaluationScore = _dspy.OutputField(desc="Continuous score in [0, 100]")


    class OverallScoreOnlySignature(_dspy.Signature):
        source: str = _dspy.InputField()
        response: str = _dspy.InputField()
        criterion_information: str = _dspy.InputField(desc="Criterion evidence or definitions in JSON")
        score_of_overall: EvaluationScore = _dspy.OutputField(desc="Continuous score in [0, 100]")


    class ActiveCriticProgram(_dspy.Module):
        """Section III-C: one McS-E call and one OS-E call (coarse: OS-E only)."""

        def __init__(
            self, instruction: TaskInstruction, mode: Mode,
            *, explanations: bool = True, aggregation: str = "llm",
        ) -> None:
            super().__init__()
            if mode not in {"coarse", "fine"}:
                raise ValueError("mode must be coarse or fine")
            if aggregation not in {"llm", "mean"} or (aggregation == "mean" and mode != "fine"):
                raise ValueError("Mean aggregation is the fine-mode w/o OS-E ablation")
            if not instruction.criteria:
                raise ValueError("Scoring requires evaluation criteria")
            self.instruction, self.mode = instruction, mode
            self.explanations, self.aggregation = explanations, aggregation
            self.demonstration_ids: list[str] = []
            task = instruction.full_task_description()
            if mode == "fine":
                template = MULTI_CRITERIA_SCORING_INSTRUCTIONS if explanations else MULTI_CRITERIA_SCORE_ONLY_INSTRUCTIONS
                self.multi_criteria = _dspy.Predict(
                    MultiCriteriaSignature.with_instructions(stage2_instructions(task, template))
                )
            if aggregation == "llm":
                signature = OverallSignature if explanations else OverallScoreOnlySignature
                if mode == "fine":
                    template = OVERALL_SCORING_INSTRUCTIONS if explanations else OVERALL_SCORE_ONLY_INSTRUCTIONS
                else:
                    template = COARSE_SCORING_INSTRUCTIONS if explanations else COARSE_SCORE_ONLY_INSTRUCTIONS
                self.overall = _dspy.Predict(signature.with_instructions(stage2_instructions(task, template)))
            self._base_instructions = {name: p.signature.instructions for name, p in self.named_predictors()}

        def set_demonstrations(self, examples: Sequence[Any]) -> None:
            """Use exactly the selected human-labeled triples in every module.

            No hidden bootstrap search, fabricated criterion labels, or extra demos.
            Store demo text in signature instructions so DSPy JSON saves preserve it.
            """
            self.demonstration_ids = [str(x.uid) for x in examples]
            if len(set(self.demonstration_ids)) != len(examples):
                raise ValueError("A demonstration subset must not repeat a training row")
            records = [
                {"Source": x.source, "Response": x.response, "Human Score": clamp_score(x.score_of_overall)}
                for x in examples
            ]
            block = "\n\nFixed human-rated demonstrations (overall human scores):\n" + json.dumps(records, ensure_ascii=False)
            for name, predictor in self.named_predictors():
                predictor.demos = []
                predictor.signature = predictor.signature.with_instructions(self._base_instructions[name] + block)

        def assess_criteria(self, *, source: str, response: str) -> dict[str, CriterionResult]:
            if self.mode != "fine":
                raise ValueError("McS-E is only available in fine mode")
            output = self.multi_criteria(
                source=source, response=response,
                multiple_evaluation_criteria=json.dumps(self.instruction.criteria, ensure_ascii=False),
            )
            return parse_criterion_assessments(
                output.score_of_each_criterion_in_json, self.instruction.criteria,
                explanations=self.explanations,
            )

        def score_overall(self, *, source: str, response: str, evidence: dict) -> tuple[str, float]:
            if self.aggregation != "llm":
                raise ValueError("The w/o OS-E ablation has no overall LLM module")
            output = self.overall(
                source=source, response=response,
                criterion_information=json.dumps(evidence, ensure_ascii=False),
            )
            explanation = str(getattr(output, "explanation", "")).strip()
            if self.explanations and not explanation:
                raise OutputFormatError("Overall evaluation omitted its explanation")
            return explanation, clamp_score(output.score_of_overall)

        def forward(self, source: str, response: str) -> Any:
            results = self.assess_criteria(source=source, response=response) if self.mode == "fine" else {}
            evidence = {
                name: (item.to_dict() if self.explanations else {"score": item.score})
                for name, item in results.items()
            } if self.mode == "fine" else self.instruction.criteria
            if self.aggregation == "mean":
                explanation, score = "", sum(x.score for x in results.values()) / len(results)
            else:
                explanation, score = self.score_overall(source=source, response=response, evidence=evidence)
            return _dspy.Prediction(
                mode=self.mode, criterion_assessments_json=json.dumps(evidence if results else {}, ensure_ascii=False),
                overall_explanation=explanation, overall_score=score,
            )
else:
    class ActiveCriticProgram:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_dspy()


def make_dspy_examples(examples: Sequence[Example]) -> list[Any]:
    dspy = require_dspy()
    return [
        dspy.Example(uid=x.uid, source=x.source, response=x.response, score_of_overall=x.human_score)
        .with_inputs("source", "response")
        for x in examples
    ]


def save_program_state(program: Any, path: str | Path) -> None:
    """DSPy serializes attached LMs including credentials; detach them during saving."""
    predictors = list(program.predictors())
    lms = [p.lm for p in predictors]
    try:
        for p in predictors:
            p.lm = None
        program.save(str(path))
    finally:
        for p, lm in zip(predictors, lms):
            p.lm = lm


class DSPyActiveCriticScorer:
    def __init__(self, program: Any, instruction: TaskInstruction, mode: Mode) -> None:
        self.program, self.instruction, self.mode = program, instruction, mode

    def score(self, example: Example) -> EvaluationResult:
        # Gold labels and reference summaries never enter evaluation inputs.
        prediction = self.program(source=example.source, response=example.response)
        criteria = parse_criterion_assessments(
            prediction.criterion_assessments_json, self.instruction.criteria,
            explanations=self.program.explanations,
        ) if self.mode == "fine" else {}
        return EvaluationResult(
            uid=example.uid, source=example.source, response=example.response,
            human_score=example.human_score, criterion_results=criteria,
            overall_explanation=prediction.overall_explanation,
            overall_score=clamp_score(prediction.overall_score),
        )
