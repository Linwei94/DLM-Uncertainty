import logging
from default_utils.custom_types import ModelOutputs, PromptCollection
from default_utils.registry import register_filter
from models.model_manager import ModelManager
import re
import ast


def output_substring_extractor(regexes: list[str], model_outputs: list[ModelOutputs]) -> list[ModelOutputs]:
    potential_answer_regex = [re.compile(rf"{rgx}") for rgx in regexes]
    def locate_substring_tokens(decoded_tokens, token_logprobs, substring):
        """Return token-level slice for the substring in decoded tokens."""
        try:
            full_text = "".join(decoded_tokens)
        except Exception:
            return decoded_tokens, token_logprobs
        start_char = full_text.find(substring)
        if start_char == -1:
            return decoded_tokens, token_logprobs

        end_char = start_char + len(substring)
        running = 0
        token_start = None
        token_end = None

        for i, tok in enumerate(decoded_tokens):
            tok_len = len(tok)

            if tok_len == 1:
                return decoded_tokens, token_logprobs

            if token_start is None and running + tok_len > start_char:
                token_start = i
            if token_end is None and running + tok_len >= end_char:
                token_end = i + 1
                break
            running += tok_len

        if token_start is None or token_end is None:
            return decoded_tokens, token_logprobs

        return decoded_tokens[token_start:token_end], token_logprobs[token_start:token_end]

    filtered_outputs = []

    for output in model_outputs:
        new_texts = []
        new_tokens = []
        new_logprobs = []

        for text, tokens, logprobs in zip(output.output_texts, output.output_tokens, output.output_logprobs):
            captured_text = None
            logging.debug(f"Original text: {text}")
            for rgx in potential_answer_regex:
                m = rgx.search(text)
                if m:
                    captured_text = m.group(1)  # capture group (answer letter)
                    logging.debug(f"Captured text: {captured_text}")
                    break

            if captured_text is None:
                new_texts.append(text)
                new_tokens.append(tokens)
                new_logprobs.append(logprobs)
                continue

            matched_tokens, matched_lp = locate_substring_tokens(tokens, logprobs, captured_text)

            new_texts.append(captured_text)
            new_tokens.append(matched_tokens)
            new_logprobs.append(matched_lp)

        filtered_outputs.append(
            ModelOutputs(
                context_texts=output.context_texts,
                output_texts=new_texts,
                output_tokens=new_tokens,
                output_logprobs=new_logprobs,
            )
        )
    print(model_outputs[0].output_texts)
    print(filtered_outputs[0].output_texts)
    return filtered_outputs


@register_filter(name="regex_extractor")
def regex_extractor(cfg: dict, model_outputs: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> list[ModelOutputs]:
    potential_answer_regex = kwargs.get("regexes", [])
    print("potential_answer_regex:", potential_answer_regex)
    return output_substring_extractor(potential_answer_regex, model_outputs)



@register_filter(name="linguistic_confidence_augmentation")
def linguistic_confidence_augmentation(cfg: dict, model_outputs: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> list[ModelOutputs]:
    aug_model_manager = ModelManager(master_cfg=cfg, model_config_type="linguistic_confidence_augmentation_model")
    all_cont_prompts = []
    for output in model_outputs:
        template = """
        You are given an answer to a question. Your task is to augment the answer by adding linguistic cues that express the confidence level of the answer.
        Please generate 4 different versions of the answer, each with a different level of confidence expressed linguistically, ranging from no confidence (e.g. "I don't know") to high confidence.
        Here are some inspirations: 
        1. no confidence: "I don't know the answer."
        2. low confidence: "I'm not sure, but I think the answer might be..."
        3. medium confidence: "I believe the answer is..."
        4. high confidence: "I'm certain that the answer is..."

        Please augment the following answer accordingly and return ONLY a JSON object with the confidence levels as keys and the corresponding augmented answers as values.
        Answer: "{answer}"

        Return format:
        {{
            "1": "...",
            "2": "...",
            "3": "...",
            "4": "..."
        }}
        """.strip()
        aug_prompt = PromptCollection(context_texts=[template.format(answer=ans) for ans in output.output_texts])
        aug_outputs: ModelOutputs = aug_model_manager.run_generation(aug_prompt)[0]
        
        lc_prompt_template = """
        Answer the following question using a succinct (at most one sentence) and full answer. 
        If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer. 
        Question: {formatted_question}
        Answer: 
        """.strip()

        for i, aug in enumerate(aug_outputs.output_texts):
            try:
                aug_dict = ast.literal_eval(aug)
                continuations = list(aug_dict.values())
                continuations = [cont for cont in continuations if cont is not None and isinstance(cont, str)]
            except:
                # hard code continuations if parsing fails
                hard_coded_aug_template = ["I don't know the answer.",
                                            "I'm not sure, but I think the answer might be: ",
                                            "I believe the answer is: ",
                                            "I'm sure that the answer is: ",]
                continuations = [temp + str(output.output_texts[i]) for temp in hard_coded_aug_template] # fallback to original answer
            prompts.context_texts[i] = lc_prompt_template.format(formatted_question=prompts.questions[i])
            prompts.continuation_texts[prompts.context_texts[i]] = continuations
        # Re-score continuations with the main model
        all_cont_prompts.append(prompts)

    qa_model = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    return [qa_model.run_continuation(p)[0] for p in all_cont_prompts]