import time
from tqdm import tqdm
from ..default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from transformers import AutoTokenizer
import gc
import numpy as np
import pickle
import os
from vllm import LLM, SamplingParams
import logging


class vLLMModel(AbstractModel):
    """
    vLLM model wrapper for general HF models.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)

    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:

        # Build chat messages from prompt collection
        messages_list = []
        for context_text in prompt_collection.context_texts:
            messages = [{"role": "system", "content": prompt_collection.system_prompt},
                        {"role": "user", "content": context_text}]
            messages_list.append(messages)
        stop_seq = self.cfg.get("stop_sequences", [])
        sampling_params = SamplingParams(temperature=self.cfg.get("temperature", 1.0),
                                         max_tokens=self.cfg.get("max_tokens", 256),
                                         logprobs=5,
                                         stop=list(stop_seq)
                                         )
        vllm_model = LLM(model=self.model_name,
                         max_model_len=self.cfg.get("max_model_len", 4096))

        model_outputs_list = []
        for _ in range(self.repeat):
            logging.info(f"vLLM [{self.model_name}] Generation Round {_ + 1}/{self.repeat}")
            # Generate responses using vLLM chat
            try:
                chat_template_kwargs={"reasoning_effort": self.cfg.get("reasoning_effort")}
                outputs = vllm_model.chat(messages_list,
                                            sampling_params=sampling_params,
                                            chat_template_kwargs=chat_template_kwargs)
            except:     
                outputs = vllm_model.generate(
                    prompt_collection.context_texts, 
                    sampling_params=sampling_params)

            # Extract output texts and tokens
            output_texts = []
            output_tokens = []
            output_logprobs = []
            all_top_k_tokens = []

            for output in outputs:
                # For each prompt, collect all n completions
                for completion in output.outputs:
                    has_assistant_token = False
                    if "assistantfinal" in completion.text:
                        has_assistant_token = True
                        generated_text = completion.text.rsplit(
                            "assistantfinal", 1)[-1].strip()
                    else:
                        generated_text = completion.text.strip()
                    output_texts.append(generated_text)

                    # Tokenize generated text to get the expected number of tokens
                    generated_token_ids = self.tokenizer.encode(
                        generated_text, add_special_tokens=False)
                    expected_length = len(generated_token_ids)

                    # Extract decoded tokens and logprobs, skipping special tokens
                    tokens = []
                    logprobs = []
                    top_ks = []
                    found_assistant = False

                    if completion.logprobs:
                        for lp in completion.logprobs:
                            if lp and len(lp) > 0:
                                # save top 1 logprob aka output logprobs
                                tok_info = list(lp.values())[0]
                                decoded_token = tok_info.decoded_token

                                # If has_assistant_token, skip tokens until we find "final"
                                match self.model_name.lower():
                                    case "gpt-oss":
                                        if has_assistant_token and not found_assistant:
                                            if "final" in decoded_token.lower():
                                                found_assistant = True
                                            continue
                                    case _:
                                        pass

                                # Skip special tokens
                                if decoded_token not in self.tokenizer.all_special_tokens and not (decoded_token.startswith("<|") and decoded_token.endswith("|>")):
                                    tokens.append(decoded_token)
                                    logprobs.append(tok_info.logprob)
                                    # save top k tokens and logprobs
                                    top_ks.append([(tk.decoded_token, tk.logprob) for tk in list(lp.values())])

                    # Slice tokens and logprobs to match generated text length
                    tokens = tokens[-expected_length:] if expected_length > 0 else tokens
                    logprobs = logprobs[-expected_length:
                                        ] if expected_length > 0 else logprobs
                    output_tokens.append(tokens)
                    output_logprobs.append(logprobs)
                    all_top_k_tokens.append(top_ks)
            model_outputs_list.append(ModelOutputs(
                context_texts=prompt_collection.context_texts,
                output_texts=output_texts,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
                top_k_tokens=all_top_k_tokens,
            ))
        vllm_model.llm_engine.engine_core.shutdown()
        del vllm_model
        del self.tokenizer
        gc.collect()
        return model_outputs_list


    def run_continuation(self, prompt_collection: PromptCollection):

        # ---- Load vLLM ----
        llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )

        tokenizer = llm.get_tokenizer()
        model_outputs_list = []

        for _ in range(self.repeat):

            all_output_texts = []
            all_output_tokens = []
            all_output_logprobs = []
            all_candidates = []

            # Iterate contexts
            for context in tqdm(prompt_collection.context_texts, desc="Scoring continuations"):
                continuations = prompt_collection.continuation_texts[context]

                candidates = []

                # ---- Build all prompts for batch scoring ----
                prompts = [context + continuation for continuation in continuations]

                # ---- Ask vLLM for logprobs along the entire output ----
                sampling = SamplingParams(
                    temperature=0,                # deterministic
                    max_tokens=1,                 # do NOT generate beyond the prompt
                    logprobs=1,                   # return token-level logprobs
                    prompt_logprobs=True,         # needed to score provided tokens
                )

                outputs = llm.generate(prompts, sampling, use_tqdm=False)

                # ---- Extract continuation logprobs ----
                for continuation, out in zip(continuations, outputs):
                    try:
                        cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
                        num_cont_toks = len(cont_ids)

                        # out.prompt_logprobs is a list of dicts, one per prompt token
                        token_logprobs = out.prompt_logprobs[-num_cont_toks:]

                        lp = []
                        tokens = []
                        for logprob_dict in token_logprobs:
                            lp.append(list(logprob_dict.values())[0].logprob)
                            tokens.append(list(logprob_dict.values())[0].decoded_token)

                        candidates.append({
                            "text": continuation,
                            "tokens": tokens,
                            "logprobs": lp,
                            "mean": float(np.mean(lp)),
                        })
                    except Exception as e:
                        logging.error(f"Error processing continuation: {continuation}\n{e}")
                        continue

                # choose highest-mean continuation
                best = max(candidates, key=lambda x: x["mean"])
                all_candidates.append(candidates)
                all_output_texts.append(best["text"])
                all_output_tokens.append(best["tokens"])
                all_output_logprobs.append(best["logprobs"])

            model_outputs_list.append(
                ModelOutputs(
                    context_texts=prompt_collection.context_texts,
                    output_texts=all_output_texts,
                    output_tokens=all_output_tokens,
                    output_logprobs=all_output_logprobs,
                    continuation_candidates=all_candidates
                )
            )
        llm.llm_engine.engine_core.shutdown()
        del llm
        del tokenizer
        del self.tokenizer
        gc.collect()
        return model_outputs_list