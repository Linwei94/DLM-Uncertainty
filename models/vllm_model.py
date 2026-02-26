import time
from tqdm import tqdm
from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from transformers import AutoTokenizer
import gc
import numpy as np
import pickle
import os
import logging

# Patch vllm Disabledtqdm for tqdm/huggingface_hub compatibility (avoid "multiple values for disable")
import vllm.model_executor.weight_utils as _weight_utils
_orig_disabled_tqdm = _weight_utils.Disabledtqdm


class _FixedDisabledtqdm(_orig_disabled_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("disable", None)  # Avoid duplicate disable= when caller passes it
        super(_orig_disabled_tqdm, self).__init__(*args, **kwargs, disable=True)


_weight_utils.Disabledtqdm = _FixedDisabledtqdm

# Patch vllm for Llama 3.x rope_scaling compatibility
import vllm.config as _vllm_config
from vllm.model_executor.layers import rotary_embedding as _rotary_embedding

_orig_get_max_len = _vllm_config._get_and_verify_max_len
_orig_get_rope = _rotary_embedding.get_rope


def _patched_get_max_len(hf_config, max_model_len):
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None and isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        raw_type = rope_scaling.get("type") or rope_scaling.get("rope_type", "linear")
        if raw_type in ("default", "llama3"):
            raw_type = "linear"
        rope_scaling["type"] = raw_type
        rope_scaling.setdefault("factor", 1.0)
        object.__setattr__(hf_config, "rope_scaling", rope_scaling)
    return _orig_get_max_len(hf_config, max_model_len)


def _patched_get_rope(*args, rope_scaling=None, **kwargs):
    """Map default/llama3 RoPE types to linear (vllm 0.2.5 only supports linear/dynamic/yarn)."""
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        raw_type = rope_scaling.get("type", "linear")
        if raw_type in ("default", "llama3"):
            rope_scaling["type"] = "linear"
        rope_scaling.setdefault("factor", 1.0)
    return _orig_get_rope(*args, rope_scaling=rope_scaling, **kwargs)


_vllm_config._get_and_verify_max_len = _patched_get_max_len
_rotary_embedding.get_rope = _patched_get_rope

from vllm import LLM, SamplingParams


class vLLMModel(AbstractModel):
    """
    vLLM model wrapper for general HF models.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer_name = self._resolve_tokenizer_name()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=True)
        self._llm = None  # Lazy init, reused across rounds

    def _resolve_tokenizer_name(self) -> str:
        """Resolve tokenizer with optional fast path for legacy LLaMA v1 models."""
        explicit = self.cfg.get("tokenizer_name", None)
        if explicit:
            return explicit

        model_lower = (self.model_name or "").lower()
        is_llama = "llama" in model_lower
        is_newer_llama = any(tag in model_lower for tag in [
            "llama-2", "llama2", "llama-3", "llama3", "llama-4", "llama4"
        ])
        use_fast_llama_v1 = self.cfg.get("use_fast_llama_v1_tokenizer", True)

        if is_llama and not is_newer_llama and use_fast_llama_v1:
            logging.info(
                "Using fast tokenizer override for LLaMA v1-style model: %s -> hf-internal-testing/llama-tokenizer",
                self.model_name,
            )
            return "hf-internal-testing/llama-tokenizer"
        return self.model_name

    def _get_llm(self):
        """Create LLM once and reuse across rounds."""
        if self._llm is None:
            # Use float16 for PyTorch 2.1 compatibility (bfloat16 can cause dtype errors)
            kwargs = dict(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                max_model_len=self.cfg.get("max_model_len", 4096),
                dtype="float16",
            )
            # Lower gpu_memory_utilization (default 0.85) to reduce OOM when GPU is shared
            kwargs["gpu_memory_utilization"] = self.cfg.get("gpu_memory_utilization", 0.85)
            self._llm = LLM(**kwargs)
        return self._llm

    def shutdown(self):
        """Release vLLM resources when done."""
        if self._llm is not None:
            try:
                self._llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            self._llm = None
            gc.collect()

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
        vllm_model = self._get_llm()

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
                                # vllm 0.2.5: lp is dict of token_id->LogProb or token_id->float
                                items = list(lp.items())
                                token_key = items[0][0]
                                tok_info = items[0][1]
                                tok_logprob = tok_info.logprob if hasattr(tok_info, "logprob") else tok_info
                                # vllm 0.2.5 uses token ids as keys; decode to string
                                decoded_token = (
                                    self.tokenizer.decode([token_key])
                                    if isinstance(token_key, int)
                                    else token_key
                                )

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
                                    logprobs.append(tok_logprob)
                                    # save top k tokens and logprobs (decode ids to strings)
                                    top_ks.append([
                                        (
                                            self.tokenizer.decode([k]) if isinstance(k, int) else k,
                                            v.logprob if hasattr(v, "logprob") else v,
                                        )
                                        for k, v in items
                                    ])

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
        # Keep LLM alive for reuse across rounds (no shutdown here)
        return model_outputs_list


    def run_continuation(self, prompt_collection: PromptCollection):

        # ---- Load vLLM ----
        llm = LLM(
            model=self.model_name,
            tokenizer=self.tokenizer_name,
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
