from default_utils.registry import register_confidence
from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
from models.model_manager import ModelManager
import re
import numpy as np


@register_confidence(name="scalar_linguistic_confidence")
def scalar_linguistic_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="linguistic_confidence_judge_model")
    def per_round_estimator(output: ModelOutputs):
        def extract_score(text: str) -> float:
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                score = min(max(score, 0.0), 100.0) / 100.0  # Normalize to [0, 1]
            else:
                score = np.nan
            return score
        
        DIRECT_PROMPT = """
        Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge). \
            Please pay attention to the hedging language used and the overall assertiveness of the statement. \
                If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. \
                    If the sentence does not contain any linguistic cues or is a succinct, decisive short answer, the score should be towards 100, too.

        Here is the sentence:
        {sentence}

        Confidence Score: [Return only a number between 0 and 100]
        """.strip()
        prompt_collection = PromptCollection(context_texts=[DIRECT_PROMPT.format(sentence=response) for response in output.output_texts])
        judge_outputs: list[ModelOutputs] = model_manager.run_generation(prompt_collection)
        confidences: list[list[float]] = [[extract_score(text) for text in out.output_texts] for out in judge_outputs]
        # Transpose confidences: from [num_judge_outputs][num_texts] to [num_texts][num_judge_outputs]
        confidences = list(map(list, zip(*confidences)))
        # Average scores from different judge outputs
        confidences = [float(np.nanmean(scores)) for scores in confidences]
        return confidences
        
    
    all_confidences = []
    all_answers = []
    for output in output_lst:
        confidences = per_round_estimator(output)
        all_confidences.append(confidences)
        all_answers.append(output.output_texts)

    return OrganisedOutputs(
        extracted_answers=all_answers,
        extracted_confidences=all_confidences,
    )