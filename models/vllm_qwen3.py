import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import gc
import numpy as np
import pickle
import os
from vllm import LLM, SamplingParams
from ..default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection


class vLLMQwen3(AbstractModel):
    """
    vLLM model wrapper for Qwen 3 models with additional thinking budget control. 
    Thinking budget control reference: https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)

    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        stop_seq = self.cfg.get("stop_sequences", [])
        vllm_model = LLM(model=self.model_name,
                         max_model_len=self.cfg.get("max_model_len", 4096))

        model_outputs_list = []
        for _ in range(self.repeat):
            logging.info(f"vLLM Qwen 3 Generation Round {_ + 1}/{self.repeat}")
            messages_list = []
            for context_text in prompt_collection.context_texts:
                messages = [{"role": "system", "content": prompt_collection.system_prompt},
                            {"role": "user", "content": context_text}]
                messages_list.append(messages)

            # thinking pass
            reasoning_effort_map = {"low": 256, "medium": 512, "high": 1024}
            thinking_budget = reasoning_effort_map.get(self.cfg.get("reasoning_effort"))
            post_thinking_messages = []
            if thinking_budget:
                logging.info(f"Qwen 3 thinking pass with budget: {thinking_budget} tokens")
                thinking_sampling_params = SamplingParams(temperature=self.cfg.get("temperature", 1.0),
                                            max_tokens=thinking_budget)
                try:
                    thinking_outputs = vllm_model.chat(messages_list, sampling_params=thinking_sampling_params)
                except:
                    thinking_budget = vllm_model.generate(prompt_collection.system_prompt, sampling_params=thinking_sampling_params)
                
                for i, thinking_output in enumerate(thinking_outputs):
                    done_thinking = 151668 in list(thinking_output.outputs[0].token_ids)
                    done_answering = 151645 in list(thinking_output.outputs[0].token_ids)
                    new_messages = messages_list[i].copy()
                    if not done_thinking:
                        # append early stopping to force end thinking
                        early_stopping_text = (
                            "\n\nConsidering the limited time by the user, "
                            "I have to give the solution based on the thinking directly now.\n"
                            "</think>\n\n"
                        )
                        new_messages[1]["content"] = new_messages[1]["content"] + "\n" + thinking_output.outputs[0].text + early_stopping_text 
                    elif not done_answering:
                        new_messages[1]["content"] = new_messages[1]["content"] + "\n" + thinking_output.outputs[0].text.rsplit("</think>")[-1].strip() + "</think>\n\n" 
                    else:
                        new_messages[1]["content"] = new_messages[1]["content"]
                    post_thinking_messages.append(new_messages)
            else:
                logging.info(f"Qwen 3 thinking skipped")
                post_thinking_messages = messages_list
            # generation pass without thinking
            sampling_params = SamplingParams(temperature=self.cfg.get("temperature", 1.0),
                                         max_tokens=self.cfg.get("max_tokens", 256),
                                         logprobs=5,
                                         stop=list(stop_seq)
                                         )
            try:
                outputs = vllm_model.chat(post_thinking_messages, 
                                        sampling_params=sampling_params, 
                                        chat_template_kwargs={"enable_thinking": False})
            except:
                non_chat_messages_list = ["/no_think " + msgs[1]["content"] for msgs in post_thinking_messages]
                outputs = vllm_model.generate(non_chat_messages_list, sampling_params=sampling_params)

            # Extract output texts and tokens
            output_texts = []
            output_tokens = []
            output_logprobs = []
            all_top_k_tokens = []   

            for output in outputs:
                # For each prompt, collect all n completions
                for completion in output.outputs:
                    generated_text = completion.text.strip()
                    output_texts.append(generated_text)
                    # Tokenize generated text to get the expected number of tokens
                    generated_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
                    expected_length = len(generated_token_ids)

                    # Extract decoded tokens and logprobs, skipping special tokens
                    tokens = []
                    logprobs = []
                    top_ks = []

                    if completion.logprobs:
                        for lp in completion.logprobs:
                            if lp and len(lp) > 0:
                                # save top 1 logprob aka output logprobs
                                tok_info = list(lp.values())[0]
                                decoded_token = tok_info.decoded_token

                                # Skip special tokens
                                if decoded_token not in self.tokenizer.all_special_tokens:
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