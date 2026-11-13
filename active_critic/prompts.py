from __future__ import annotations


TASK_OBSERVATION_PROMPT = """Given several examples from an NLG evaluation dataset where each entry consists of a ‘Source’ text and its corresponding ‘Response’, and, when available, a Human Score that evaluates the response quality. When scores are absent, use only the supplied texts and do not invent human feedback.
Please write observations about trends that hold for most or all of the samples. I will also provide you with some previous observations I have already made. Please add your observations or if you feel the observations are comprehensive say ‘COMPLETE’.
Some areas you may consider in your observations: content and structure, scenario, task, evaluation objective, evaluation criteria, etc.

It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don’t be afraid to be creative.
{examples}
{prior_observations}"""


TASK_DESCRIPTION_PROMPT = """Given a series of observations I have made and some description about this NLG evaluation dataset.
1. Identify the type of evaluation task. Possible tasks include: machine translation, text summarization, data-to-text generation, dialogue generation, image description, text simplification, story generation, paraphrase generation, textual entailment, reasoning, etc.
2. What this evaluation task refers to evaluating.
3. Output the relationship between ‘Source’ and ‘Response’ in this task in 1-3 sentences.
4. Given a summary in fill [ ]: The evaluation task is [ ], which refers to evaluating [ ] generated from [ ]. The relationship between ‘Source’ and ‘Response’ is [ ].
{observations}
{prior_task_description}"""


TASK_ENSEMBLE_PROMPT = """Synthesize ALL the following mini-batch task descriptions into one final task description. Resolve inconsistencies using the patterns shared across the descriptions. Preserve the evaluation objective and the relationship between Source and Response.
Return this format:
The evaluation task is [task type], which refers to evaluating [outputs] generated from [inputs]. The relationship between 'Source' and 'Response' is [relationship].
Mini-batch task descriptions:
{task_descriptions}"""


INITIAL_CRITERIA_PROMPT = """Task description:
{task_description}
Given several examples from {task_type} evaluation dataset. Each example contains a Source text and a generated Response; Human Score is included when feedback is available.
Please write observations about the relationship between response quality and human scores when feedback is available, and infer which evaluation criteria the human annotators most likely focused on. If scores are absent, infer plausible criteria from the texts without inventing human judgments.
Analyze the trends across examples. Return the observed patterns and main evaluation criteria. Make each criterion's preferred behavior explicit.
Examples:{examples}
Output in the following format:
{{
"Criteria": {{
"Criterion A": "Short description of Criterion A",
"Criterion B": "Short description of Criterion B",
... }},
"Observations": "Observed patterns and insights about how human annotators likely evaluated the responses based on the examples." }}"""


CRITERIA_OBSERVATION_PROMPT = """Task description:
{task_description}
Given several examples from {task_type} evaluation dataset. Each example contains a Source text and a generated Response; Human Score is included when feedback is available. If scores are absent, use only the supplied texts without inventing human judgments. Your task is to review the existing observations and evaluation criteria.
I will provide you with: - A batch of new examples - Previously generated observations - A current set of evaluation criteria believed to reflect human scoring.
Please assess whether:
1. The existing observations sufficiently capture patterns in the data.
2. The evaluation criteria accurately represent the dimensions human annotators focus on when assigning scores.
If you feel the observations and criteria are already comprehensive and accurate, say ‘COMPLETE’.
Otherwise, provide a paragraph of new insights or refinements that should be considered.
{examples}
{prior_observations}
{current_evaluation_criteria}"""


REFINE_CRITERIA_PROMPT = """Task description:
{task_description}
Given a set of evaluation criteria and new observations.
Your task is to refine the current evaluation criteria based on the new observations.
Output a refined set of {min_criteria} to {max_criteria} evaluation criteria in JSON format: criterion as key, description as value.
{evaluation_criteria}
{new_observations}"""


# The explanation ablation keeps scoring instructions, inputs, and score scale
# fixed. Only explanation generation and its corresponding output fields differ.
_MULTI_CRITERIA_BASE = """Evaluate the response with respect to the source on ALL the supplied evaluation criteria, simultaneously in one call. Use exactly the supplied criterion names and definitions.
For each criterion, return a continuous score from 0 to 100. 0 means completely failing that criterion; 100 means fully satisfying it."""
_DEMONSTRATION_NOTE = """The human-rated demonstrations establish the scoring preferences. Their human scores are overall scores, not labels for individual criteria."""

MULTI_CRITERIA_SCORING_INSTRUCTIONS = _MULTI_CRITERIA_BASE + """
Include a brief explanation for each criterion. Return a JSON object with one entry per criterion:
{"criterion name": {"explanation": "reason for this criterion score", "score": 75.5}}.
""" + _DEMONSTRATION_NOTE

MULTI_CRITERIA_SCORE_ONLY_INSTRUCTIONS = _MULTI_CRITERIA_BASE + """
Return a JSON object with one entry per criterion:
{"criterion name": {"score": 75.5}}.
Return scores only, without explanations.
""" + _DEMONSTRATION_NOTE

_OVERALL_FINE_BASE = """Rate the overall quality of the response using the source, task description, human-rated demonstrations, and the supplied criterion assessments. Synthesize the intermediate assessments into a continuous overall score from 0 to 100, where 0 means completely failing the evaluation criteria and 100 means fully satisfying them."""
_OVERALL_COARSE_BASE = """Rate the overall quality of the response using the source, task description, human-rated demonstrations, and ALL supplied criterion names and definitions together. Return a continuous overall score from 0 to 100, where 0 means completely failing the evaluation criteria and 100 means fully satisfying them. Do not generate separate criterion scores."""
_EXPLAIN_OVERALL = """
Explain the overall judgment, including relevant errors and possible improvements. Return the explanation and the overall score."""
_SCORE_ONLY_OVERALL = """
Return the overall score only. Do not generate an explanation."""

OVERALL_SCORING_INSTRUCTIONS = _OVERALL_FINE_BASE + _EXPLAIN_OVERALL
OVERALL_SCORE_ONLY_INSTRUCTIONS = _OVERALL_FINE_BASE + _SCORE_ONLY_OVERALL
COARSE_SCORING_INSTRUCTIONS = _OVERALL_COARSE_BASE + _EXPLAIN_OVERALL
COARSE_SCORE_ONLY_INSTRUCTIONS = _OVERALL_COARSE_BASE + _SCORE_ONLY_OVERALL


def stage2_instructions(task_description: str, template: str) -> str:
    return f"{task_description.strip()}\n{template}"
