from ..default_utils.custom_types import OrganisedOutputs, PromptCollection, ModelOutputs
from ..default_utils.datasets_manager import DatasetsManager
from ..default_utils.registry import register_grader
from ..models.model_manager import ModelManager
import pandas as pd

@register_grader(name="exact_match")
def exact_match(cfg: dict, extracted_output: OrganisedOutputs, prompts: PromptCollection, dataset_manager: DatasetsManager = None):
    exact_matches = []
    answer_keys = prompts.answer_keys
    for round_outputs in extracted_output.extracted_answers:
        round_matches = []
        for pred, ref in zip(round_outputs, answer_keys):
            try:
                if pred is None:
                    round_matches.append(None)
                # take first character match as correct
                elif pred.strip().upper()[0] == ref.strip().upper()[0]:
                    round_matches.append(1)
                else:
                    round_matches.append(0)
            except:
                round_matches.append(0)
        exact_matches.append(round_matches)
    return exact_matches


# general LLM-based grader
@register_grader(name="llm_grader")
def llm_grader(cfg: dict, extracted_output: OrganisedOutputs, prompts: PromptCollection, dataset_manager: DatasetsManager = None):
    model = ModelManager(master_cfg=cfg, model_config_type="grader_model")
    correct_answers = prompts.answer_keys
    questions = prompts.context_texts
    all_scores = []
    for round_outputs in extracted_output.extracted_answers:
        grading_prompts = PromptCollection(context_texts=[], continuation_texts={})
        round_scores = []
        for question, predicted_answer, correct_answer in zip(questions, round_outputs, correct_answers):
            grading_prompt = f"""
            Your job is to look at a question with a correct answer and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
            If the predicted answer matches, implies or covers the correct answer, the grade is CORRECT.
            If the predicted answer does not match, imply or cover the correct answer, the grade is INCORRECT. Do NOT grade it as INCORRECT if the predicted answer abstain from answering (e.g. "I don't know the answer..." or "I have no idea...").
            If the predicted answer is empty, none or abstention (e.g. "I don't know the answer..." or "I have no idea..."), grade the predicted answer as NOT_ATTEMPTED instead of CORRECT or INCORRECT. If the predicted answer makes an attempt (even random guesses), do not grade it as NOT_ATTEMPTED.
            Ignore any explanation or linguistic cues present in the predicted answer. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
            
            ```
            Question: {question}
            Correct answer: {correct_answer}
            Predicted answer: {"" if predicted_answer is None else predicted_answer}
            ```

            Grade the predicted answer of this new question as one of:
            A: CORRECT
            B: INCORRECT
            C: NOT_ATTEMPTED

            Just return one of the letters "A", "B", or "C", with no text around it.
            """.strip()
            grading_prompts.context_texts.append(grading_prompt)
            grading_prompts.continuation_texts[grading_prompt] = ["A", "B", "C"]
        
        for output_text in model.run_generation(grading_prompts)[0].output_texts:
            if "B" == output_text.upper().strip() or "INCORRECT" == output_text.upper().strip():
                round_scores.append(0)
            elif "A" == output_text.upper().strip() or "CORRECT" == output_text.upper().strip():
                round_scores.append(1)
            elif "C" == output_text.upper().strip() or "NOT_ATTEMPTED" == output_text.upper().strip():
                round_scores.append("")
            else:
                round_scores.append(None)
        all_scores.append(round_scores)
    return all_scores
