from vllm import LLM, SamplingParams
from tqdm import tqdm  # optional progress bar
import numpy as np

def sample(model_id, prompts, repeat=1, max_tokens=1024, temperature=0, show_progress=True):
    llm = LLM(model=model_id, max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=max_tokens, logprobs=1, temperature=temperature)

    n_prompts = len(prompts)
    all_responses = [[] for _ in range(n_prompts)]
    all_tokens = [[] for _ in range(n_prompts)]
    all_logprobs = [[] for _ in range(n_prompts)]

    iterator = range(repeat)
    if show_progress:
        iterator = tqdm(iterator, desc="Sampling rounds")

    for _ in iterator:
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            logprob_info = output.outputs[0].logprobs  # list of dicts

            tokens, logprobs = [], []
            for lp in logprob_info:
                if lp and len(lp) > 0:
                    tok_info = list(lp.values())[0]
                    tokens.append(tok_info.decoded_token)
                    logprobs.append(tok_info.logprob)
                else:
                    tokens.append(None)
                    logprobs.append(None)

            all_responses[i].append(text)
            all_tokens[i].append(tokens)
            all_logprobs[i].append(logprobs)

    return all_responses, all_tokens, all_logprobs


def p_true_eval(model_id, prompts, repeat=1, max_tokens=8, temperature=0):

    # True/False token variants
    tf_variants = {
        "True": ["TRUE", "True", "true", "T", "t", "ĠTrue", "Ġtrue", "Yes", "yes", "ĠYes", "Ġyes"],
        "False": ["FALSE", "False", "false", "F", "f", "ĠFalse", "Ġfalse", "No", "no", "ĠNo", "Ġno"]
    }

    llm = LLM(model=model_id, max_model_len=2048)
    sampling_params = SamplingParams(max_tokens=max_tokens, logprobs=20, temperature=temperature)

    n_prompts = len(prompts)

    confs = []

    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        logprob_info = output.outputs[0].logprobs  # list of dicts
        print(logprob_info)
        tokens, logprobs = [], []
        temp_map = {key: 0.0 for key in tf_variants.keys()}

        for lp in logprob_info: # a list of dict[id: (logprob, rank, decoded_token)], each dict corresponds to a generated token
            if list(lp.values())[0].decoded_token.strip().lower() in tf_variants["True"] or list(lp.values())[0].decoded_token.strip().lower() in tf_variants["False"]:
                for top_k in lp.values():
                    if top_k.decoded_token.strip().lower() in tf_variants["True"]:
                        temp_map["True"] += float(np.exp(top_k.logprob))
                    elif top_k.decoded_token.strip().lower() in tf_variants["False"]:
                        temp_map["False"] += float(np.exp(top_k.logprob))
                break

        print(temp_map)
        try:
            p_true = temp_map["True"] / sum(temp_map.values())
            confs.append(p_true)
        except:
            confs.append(None)

    return confs

def sample_4_choices(model_id, prompts, repeat=1, max_tokens=8, temperature=0):
    # Define A/B/C/D token variants
    opt_variants = {
        "A": ["a", "A"],
        "B": ["b", "B"],
        "C": ["c", "C"],
        "D": ["d", "D"]
    }

    llm = LLM(model=model_id, max_model_len=2048)
    sampling_params = SamplingParams(max_tokens=max_tokens, logprobs=20, temperature=temperature)

    outputs, all_tokens, all_logprobs = [], [], []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        prompt_outputs, per_prompt_tokens, per_prompt_logprobs = [], [], []

        for _ in range(repeat):
            llm_outputs = llm.generate([prompt], sampling_params)
            output = llm_outputs[0]
            text = output.outputs[0].text.strip()
            logprob_info = output.outputs[0].logprobs  # list[dict] per generated token

            tokens = []
            temp_map = {key: 0.0 for key in opt_variants.keys()}
            found_choice = None

            # ---- find first occurrence of A/B/C/D ----
            for lp in logprob_info:
                if not lp:
                    continue
                tok = list(lp.values())[0].decoded_token.strip()

                # is this token one of A/B/C/D?
                for opt, variants in opt_variants.items():
                    if tok in variants:
                        found_choice = opt
                        # accumulate probability mass for all four options at this position
                        for top_k in lp.values():
                            tk_decoded = top_k.decoded_token.strip()
                            for sub_opt, sub_variants in opt_variants.items():
                                if tk_decoded in sub_variants:
                                    temp_map[sub_opt] += float(np.exp(top_k.logprob))
                        break
                if found_choice is not None:
                    break

            # ---- compute normalized log probability ----
            try:
                total_p = sum(temp_map.values())
                if found_choice is not None and total_p > 0:
                    print(found_choice, temp_map)
                    log_p = float(np.log(temp_map[found_choice] / total_p))
                else:
                    log_p = None
            except:
                log_p = None
            if found_choice:
                prompt_outputs.append(found_choice)
                per_prompt_tokens.append([found_choice])
                per_prompt_logprobs.append(log_p)
            else:
                prompt_outputs.append(text)
                per_prompt_tokens.append([t.decoded_token for lp in logprob_info if lp for t in lp.values()])
                per_prompt_logprobs.append(log_p)

        outputs.append(prompt_outputs)
        all_tokens.append(per_prompt_tokens)
        all_logprobs.append(per_prompt_logprobs)

    return outputs, all_tokens, all_logprobs



# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# print(sample_4_choices("meta-llama/Meta-Llama-3-8B-Instruct", ["""
# Question: If a psychologist acts as both a fact witness for the plaintiff and an expert witness for the court in a criminal trial, she has acted:
# A. unethically by accepting dual roles.
# B. ethically as long as she did not have a prior relationship with the plaintiff.
# C. ethically as long as she clarifies her roles with all parties.
# D. ethically as long as she obtains a waiver from the court.
# Answer: [Return Your Answer Letter ONLY]
# """]), )