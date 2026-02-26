from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
import numpy as np
from default_utils.registry import register_confidence
from models.model_manager import ModelManager


@register_confidence(name="verbalised_numerical_confidence_with_llm_extractor")
def verbalised_numerical_confidence_with_llm_extractor(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # for each vnc response, extract the numerical confidence value from the text
    import json
    from vllm import LLM, SamplingParams
    # Load the model once per round
    model_id = "openai/gpt-oss-20b"
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=4096)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
    )
    def per_round_estimator(outputs: ModelOutputs) -> list[float]:
        prompts_llm = []
        for text, context in zip(outputs.output_texts, outputs.context_texts):
            prompt = f"""
            You are a strict information extractor. You are given a response to a question.
            The response text is an answer followed by a confidence score. Extract the answer verbatim, \
                without using any outside knowledge or summarisation. Extract the score as presented. \
                    Return a valid JSON object strictly in this format:
            {{
                "answer": "<the answer extracted from the text; if the questions is a multiple choice, the response may be a choice letter>",
                "confidence_score": <a number between 0 and 100 estimating confidence, verbalised in the text. Return `None` if not present.>
            }}

            Question:
            {context}

            Response text:
            {text}

            Return only the JSON object as specified above, without any additional text.
            """
            prompts_llm.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.strip()}
            ])
        # ---- Call the LLM ----
        outputs_llm = llm.chat(
            prompts_llm,
            sampling_params,
            chat_template_kwargs={"reasoning_effort": "low"}
        )

        # ---- Extract JSON ----
        cleaned = []
        for output in outputs_llm:
            text = output.outputs[0].text.strip().rsplit("assistantfinal", 1)[-1]
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(text[start:end+1])
                    cleaned.append(parsed)
                    continue
                except Exception:
                    pass
            cleaned.append({})

        # ---- Convert to answer + confidence ----
        extracted_scores = []
        extracted_answers = []

        for i, x in enumerate(cleaned):
            if isinstance(x, dict):
                extracted_answers.append(x.get("answer", outputs.output_texts[i]))

                score = x.get("confidence_score")
                if score is None:
                    extracted_scores.append(None)
                else:
                    try:
                        extracted_scores.append(float(score) / 100.0)
                    except:
                        extracted_scores.append(None)
            else:
                extracted_answers.append(outputs.output_texts[i])
                extracted_scores.append(None)

        return extracted_answers, extracted_scores

    # Apply estimator to each ModelOutputs object
    all_answers_and_scores = [per_round_estimator(o) for o in output_lst]
    all_answers = [ans for ans, _ in all_answers_and_scores]
    all_scores = [scores for _, scores in all_answers_and_scores]
    return OrganisedOutputs(extracted_answers=all_answers, extracted_confidences=all_scores)



@register_confidence(name="verbalised_numerical_confidence_by_continuation")
def verbalised_confidence_by_continuation(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # for each vnc response, score the continuation of confidence values from 0 to 100 in steps of 5
    def per_round_estimator(outputs: ModelOutputs) -> list[float]:
        continuation_prompt_collection = PromptCollection()
        for formatted_question, response in zip(prompts.questions, outputs.output_texts):
            prompt = f"""
            Read the question, provide your answer and your confidence in this answer. 
            Note: The confidence indicates how likely you think your answer is true. 
            Use the following format to answer: 
            
            ```Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]``` 

            Question:{formatted_question} 
            
            Now, please answer this question and provide your confidence level.

            """.strip()
            continuation_prompt_collection.context_texts.append(prompt)
            continuation_prompt_collection.continuation_texts[prompt] = [f"Answer and Confidence (0-100): {response}, {confidence}" for confidence in range(0, 101, 10)]
        
        model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="qa_model")
        vnc_continuation_outputs: ModelOutputs = model_manager.run_continuation(continuation_prompt_collection)[0]
        # extract the confidence of each continuation
        extracted_scores = []
        extracted_answers = []
        for max_prob_vnc in vnc_continuation_outputs.output_texts:
            max_prob_vnc = max_prob_vnc.replace("Answer and Confidence (0-100): ", "").split(", ")[1].strip("%")
            extracted_answers = outputs.output_texts
            extracted_scores.append(float(max_prob_vnc) / 100.0)
        return extracted_answers, extracted_scores
    
    # Apply estimator to each ModelOutputs object
    all_answers_and_scores = [per_round_estimator(o) for o in output_lst]
    all_answers = [ans for ans, _ in all_answers_and_scores]
    all_scores = [scores for _, scores in all_answers_and_scores]
    print(all_scores)
    return OrganisedOutputs(extracted_answers=all_answers, extracted_confidences=all_scores)