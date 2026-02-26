from ..default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
from ..models.model_manager import ModelManager
from ..default_utils.registry import register_confidence
import numpy as np
import logging
import re


@register_confidence(name="p_true_by_continuation")
def p_true_by_continuation(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    model = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    true_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})
    p_true_prompt_template = """
    Question: {question}
    Proposed Answer: {model_answer}
    Is the proposed answer:
    (A) True
    (B) False
    Return only A or B. The proposed answer is:
    """.strip()
    def per_round_estimator(outputs: ModelOutputs) -> list[float]:
        for question, model_answer in zip(outputs.context_texts, outputs.output_texts):
            if model_answer:
                eval_prompt = p_true_prompt_template.format(question=question, model_answer=model_answer)
            else:
                eval_prompt = p_true_prompt_template.format(question=question, model_answer="No answer provided.")
            true_continuation.context_texts.append(eval_prompt)
            true_continuation.continuation_texts[eval_prompt] = [" A"]
        logging.info("Scoring P(True) 'True' continuation token")
        true_results: ModelOutputs = model.run_continuation(true_continuation)[0]
        extracted_true_probs: list[float] = [float(np.exp(np.mean(logprobs))) for logprobs in true_results.output_logprobs]
        return extracted_true_probs
    
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )


@register_confidence(name="p_true_by_monte_carlo_generation")
def p_true_by_monte_carlo_generation(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    p_true_prompt_template = """
    Question: {question}
    Possible Answer: {model_answer}
    Is the possible answer:
    (A) True
    (B) False
    Return only (A) or (B). The possible answer is:
    """.strip()
    p_true_cfg = cfg.copy()
    model = ModelManager(master_cfg=p_true_cfg, model_config_type="p_true_mc_model")
    def per_round_estimator(outputs: ModelOutputs) -> list[float]:
        # Build prompts fresh per round to avoid leaking state across evaluations
        ctx_texts = []
        for question, model_answer in zip(outputs.context_texts, outputs.output_texts):
            if model_answer:
                eval_prompt = p_true_prompt_template.format(question=question, model_answer=model_answer)
            else:
                eval_prompt = p_true_prompt_template.format(question=question, model_answer="No answer provided.")
            ctx_texts.append(eval_prompt)
        p_true_prompt_collection = PromptCollection(context_texts=ctx_texts)
        logging.info("Running P(True) by Monte Carlo generation") 
        p_true_results: list[ModelOutputs] = model.run_generation(p_true_prompt_collection)
        # Each ModelOutputs in p_true_results holds responses for the same set of
        # questions; accumulate counts per question across all rounds.
        num_questions = len(outputs.context_texts)
        counts_a = [0] * num_questions
        counts_b = [0] * num_questions

        for result in p_true_results:
            logging.debug(f"Monte Carlo generation result: {result}")
            for idx, output in enumerate(result.output_texts):
                # Extract last occurrence of ANSWER: (A)/(B) or ANSWER IS: (A)/(B)
                match = None
                for pattern in [r'ANSWER\s*IS\s*:*\s*\(*([AB])\)*', r'ANSWER\s*:*\s*\(*([AB])\)*', r'\s*\(*([AB])\)*']:
                    matches = list(re.finditer(pattern, output.upper()))
                    if matches:
                        match = matches[-1]  # Get last occurrence
                        break
                
                if match:
                    choice = match.group(1)
                    if choice == 'A':
                        counts_a[idx] += 1
                    elif choice == 'B':
                        counts_b[idx] += 1
                else:
                    # Fallback to simpler token matching
                    upper = output.upper()
                    if any(token in upper for token in ["(A)", "A", "TRUE"]):
                        counts_a[idx] += 1
                    elif any(token in upper for token in ["(B)", "B", "FALSE"]):
                        counts_b[idx] += 1

        extracted_true_probs: list[float] = []
        for a_count, b_count in zip(counts_a, counts_b):
            try:
                total = a_count + b_count
                p_true = a_count / total if total > 0 else 0.0
                extracted_true_probs.append(p_true)
            except:
                extracted_true_probs.append(0.0)
        return extracted_true_probs
    
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )
