import os
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os 
import shutil
import pandas as pd
from tqdm import tqdm

GEN_LENGTH = 128

# source path (your modified file)
src = "/home/ivan/DLM-Uncertainty/models/custom_dream_generation_utils.py"
# destination path (cached Hugging Face location)
dst_dir = "/hdd/.cache/huggingface/modules/transformers_modules/Dream-org/Dream-v0-Instruct-7B/05334cb9faaf763692dcf9d8737c642be2b2a6ae"
dst = os.path.join(dst_dir, "generation_utils.py")
# make sure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)
# copy the file
shutil.copy2(src, dst)



def sample(model, prompts, repeat=1, gen_length=GEN_LENGTH, temperature=0., alg='entropy'):
    model_path = model
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()

    outputs = []
    all_tokens = []
    all_logprobs = []

    for prompt in tqdm(prompts, desc=f"Processing prompts"):
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True, padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        prompt_outputs = []
        per_prompt_tokens = []
        per_prompt_logprobs = []

        for _ in range(repeat):
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                output_history=True,
                return_dict_in_generate=True,
                steps=gen_length,
                temperature=temperature,
                top_p=0.95,
                alg=alg,
                alg_temp=0.,
            )

            generations = [
                tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(input_ids, output["sequences"])
            ]
            new_token_ids = [g[len(input_ids[0]):] for g in output["sequences"]][0]
            new_tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
            conf_list = output["confidence"]  # only generated tokens

            filtered_tokens_list = []
            filtered_conf_list = []
            filtered_top_k_tokens = []
            filtered_top_k_probs = []

            top_k_tokens = output["top_k_tokens_ids"][0][-len(new_tokens):]
            top_k_probs = output["top_k_tokens_probs"][0][-len(new_tokens):]

            for t, c, top_k_t, top_k_prob in zip(new_tokens, conf_list, top_k_tokens, top_k_probs):
                if t not in tokenizer.all_special_tokens:
                    filtered_tokens_list.append(t)
                    filtered_conf_list.append(c)
                    filtered_top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k_t))
                    filtered_top_k_probs.append(np.log(top_k_prob).tolist())

            print(filtered_top_k_tokens)
            print(filtered_top_k_probs)

            prompt_outputs.append(generations[0].split(tokenizer.eos_token)[0])
            per_prompt_tokens.append(filtered_tokens_list)
            per_prompt_logprobs.append(filtered_conf_list)

        outputs.append(prompt_outputs)
        all_tokens.append(per_prompt_tokens)
        all_logprobs.append(per_prompt_logprobs)

    return outputs, all_tokens, all_logprobs


def sample_4_choices(model, prompts, repeat=1, gen_length=3, temperature=0., alg='entropy'):
    model_path = model
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()

    # Define token variants
    option_variants = {
        "A": ["A", "a", "ĠA", "Ġa", "ĠAs", "Ġas"],
        "B": ["B", "b", "ĠB", "Ġb", "ĠBs", "Ġbs"],
        "C": ["C", "c", "ĠC", "Ġc", "ĠCs", "Ġcs"],
        "D": ["D", "d", "ĠD", "Ġd", "ĠDs", "Ġds"]
    }

    outputs = []
    all_tokens = []
    all_logprobs = []

    for prompt in tqdm(prompts, desc=f"Processing prompts"):
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True, padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        prompt_outputs = []
        per_prompt_tokens = []
        per_prompt_logprobs = []

        for _ in range(repeat):
            temp_map = {key: 0 for key in option_variants.keys()}

            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                output_history=True,
                return_dict_in_generate=True,
                steps=gen_length,
                temperature=temperature,
                top_p=0.95,
                alg=alg,
                alg_temp=0.,
            )

            generations = [
                tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(input_ids, output["sequences"])
            ]
            new_token_ids = [g[len(input_ids[0]):] for g in output["sequences"]][0]
            new_tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
            conf_list = output["confidence"]  # only generated tokens

            filtered_tokens_list = []
            filtered_conf_list = []
            filtered_top_k_tokens = []
            filtered_top_k_probs = []

            top_k_tokens = output["top_k_tokens_ids"][0][-len(new_tokens):]
            top_k_probs = output["top_k_tokens_probs"][0][-len(new_tokens):]

            for t, c, top_k_t, top_k_prob in zip(new_tokens, conf_list, top_k_tokens, top_k_probs):
                if t not in tokenizer.all_special_tokens:
                    filtered_tokens_list.append(t)
                    filtered_conf_list.append(c)
                    filtered_top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k_t))
                    filtered_top_k_probs.append(np.log(top_k_prob).tolist())

            for pos, token in enumerate(filtered_tokens_list):
                matched = False
                for opt, variants in option_variants.items():
                    if token in variants:
                        matched = True

                        # Get the top-k tokens and probs for this matched position only
                        tk_tokens = filtered_top_k_tokens[pos]
                        tk_probs = filtered_top_k_probs[pos]

                        # Accumulate probability for all options if their variants appear in top-k
                        for sub_opt, sub_variants in option_variants.items():
                            for variant in sub_variants:
                                if variant in tk_tokens:
                                    idx = tk_tokens.index(variant)
                                    prob = tk_probs[idx]
                                    temp_map[sub_opt] += np.exp(prob)  # undo log to sum probs

                        break  # stop checking other options for this token

                if matched:
                    break  # stop scanning further tokens; only first ABCD token matters

            # Compute normalized log-probabilities
            try:
                total_prob = sum(temp_map.values())
                if total_prob > 0:
                    per_prompt_logprobs.append(float(np.log(temp_map[opt] / total_prob)))
                else:
                    per_prompt_logprobs.append(None)
            except:
                per_prompt_logprobs.append(None)
                
            prompt_outputs.append(generations[0].split(tokenizer.eos_token)[0])
            per_prompt_tokens.append(filtered_tokens_list)

        outputs.append(prompt_outputs)
        all_tokens.append(per_prompt_tokens)
        all_logprobs.append(per_prompt_logprobs)

    return outputs, all_tokens, all_logprobs



def p_true_eval(model, prompts, repeat=1, gen_length=3, temperature=0., alg='entropy'):
    model_path = model
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()

    # Define token variants
    option_variants = {
        "True": ["True", "true", "ĠTrue", "Ġtrue", "Yes", "yes", "ĠYes", "Ġyes"],
        "False": ["False", "false", "ĠFalse", "Ġfalse", "No", "no", "ĠNo", "Ġno"]
    }

    outputs = []
    all_tokens = []
    all_logprobs = []

    for prompt in tqdm(prompts, desc=f"Processing prompts"):
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True, padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        prompt_outputs = []
        per_prompt_tokens = []
        per_prompt_logprobs = []

        for _ in range(repeat):
            temp_map = {key: 0 for key in option_variants.keys()}

            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                output_history=True,
                return_dict_in_generate=True,
                steps=gen_length,
                temperature=temperature,
                top_p=0.95,
                alg=alg,
                alg_temp=0.,
            )

            generations = [
                tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(input_ids, output["sequences"])
            ]
            new_token_ids = [g[len(input_ids[0]):] for g in output["sequences"]][0]
            new_tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
            conf_list = output["confidence"]  # only generated tokens

            filtered_tokens_list = []
            filtered_conf_list = []
            filtered_top_k_tokens = []
            filtered_top_k_probs = []

            top_k_tokens = output["top_k_tokens_ids"][0][-len(new_tokens):]
            top_k_probs = output["top_k_tokens_probs"][0][-len(new_tokens):]

            for t, c, top_k_t, top_k_prob in zip(new_tokens, conf_list, top_k_tokens, top_k_probs):
                if t not in tokenizer.all_special_tokens:
                    filtered_tokens_list.append(t)
                    filtered_conf_list.append(c)
                    filtered_top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k_t))
                    filtered_top_k_probs.append(np.log(top_k_prob).tolist())

            for pos, token in enumerate(filtered_tokens_list):
                matched = False
                for opt, variants in option_variants.items():
                    if token in variants:
                        matched = True

                        # Get the top-k tokens and probs for this matched position only
                        tk_tokens = filtered_top_k_tokens[pos]
                        tk_probs = filtered_top_k_probs[pos]

                        # Accumulate probability for all options if their variants appear in top-k
                        for sub_opt, sub_variants in option_variants.items():
                            for variant in sub_variants:
                                if variant in tk_tokens:
                                    idx = tk_tokens.index(variant)
                                    prob = tk_probs[idx]
                                    temp_map[sub_opt] += np.exp(prob)  # undo log to sum probs

                        break  # stop checking other options for this token

                if matched:
                    break  # stop scanning further tokens; only first ABCD token matters

            # Compute normalized log-probabilities
            try:
                total_prob = sum(temp_map.values())
                if total_prob > 0:
                    per_prompt_logprobs.append(float(np.log(temp_map[opt] / total_prob)))
                else:
                    per_prompt_logprobs.append(None)
            except:
                per_prompt_logprobs.append(None)
            
            print(temp_map)
            prompt_outputs.append(generations[0].split(tokenizer.eos_token)[0])
            per_prompt_tokens.append(filtered_tokens_list)

        outputs.append(prompt_outputs)
        all_tokens.append(per_prompt_tokens)
        all_logprobs.append(per_prompt_logprobs)

    return outputs, all_tokens, all_logprobs



# prompt = """
# Question: Who was the first president of the United States?
# Proposed Answer: George Washington
# Is the proposed answer:
#     True
#     False
# Output either True or False with no other text around it.
# """

# print(p_true_eval("Dream-org/Dream-v0-Instruct-7B", [prompt]))