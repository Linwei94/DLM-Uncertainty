import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.inference_mode()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    token_probs = None 

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            if token_probs is None:
                token_probs = confidence
            else:
                mask = torch.isfinite(confidence)
                token_probs = torch.where(mask, confidence, token_probs)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    topk_probs, topk_ids = torch.topk(F.softmax(logits_with_noise, dim=-1), k=100, dim=-1)
    return x, token_probs, topk_ids, topk_probs 


def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
    

def sample(model, prompts, repeat=1, steps=10, gen_length=10, block_length=10, temperature=0., cfg_scale=0., remasking='low_confidence'):
    
    # device = 'cuda'

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )
        
    outputs = []
    all_tokens = []
    all_logprobs = []
    for prompt in tqdm(prompts, desc=f"Processing prompts"):
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        prompt_outputs = []
        per_prompt_tokens = []
        per_prompt_logprobs = []
        for _ in range(repeat):
            with torch.inference_mode():
                out, conf, top_k_ids, top_k_probs = generate(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, cfg_scale=cfg_scale, remasking=remasking)

                out = out.cpu()
                # Get the special token IDs
                special_token_ids = set(tokenizer.all_special_ids)
                # Convert output IDs to numpy
                out_ids = out[:, input_ids.shape[1]:].cpu().numpy()
                # Convert confidence tensor
                conf_np = conf.detach().to(torch.float32).cpu().numpy()[:, input_ids.shape[1]:]
                # Build a boolean mask for non-special tokens and finite confidence values
                valid_mask = np.isfinite(conf_np) & ~np.isin(out_ids, list(special_token_ids))
                # Apply the mask
                filtered_conf = conf_np[valid_mask]
                logprobs = np.log(filtered_conf).tolist()

                response = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                tokens = tokenizer.convert_ids_to_tokens(out[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

                prompt_outputs.append(response)
                per_prompt_tokens.append(tokens)
                per_prompt_logprobs.append(logprobs)
                torch.cuda.empty_cache()
        outputs.append(prompt_outputs)
        all_tokens.append(per_prompt_tokens)
        all_logprobs.append(per_prompt_logprobs)
        del out, conf
        torch.cuda.empty_cache() 

    return outputs, all_tokens, all_logprobs


def sample_4_choices(model, prompts, repeat=1, steps=3, gen_length=3, block_length=3, temperature=0., cfg_scale=0., remasking='low_confidence'):
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )
        
    outputs = []
    all_tokens = []
    all_logprobs = []

    # Define token variants
    option_variants = {
        "A": ["A", "a", "ĠA", "Ġa", "ĠAs", "Ġas"],
        "B": ["B", "b", "ĠB", "Ġb", "ĠBs", "Ġbs"],
        "C": ["C", "c", "ĠC", "Ġc", "ĠCs", "Ġcs"],
        "D": ["D", "d", "ĠD", "Ġd", "ĠDs", "Ġds"]
    }

    for prompt in tqdm(prompts, desc=f"Processing prompts"):
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        prompt_outputs = []
        per_prompt_tokens = []
        per_prompt_logprobs = []
        # per_prompt_option_probs = []

        for _ in range(repeat):
            temp_map = {key: 0 for key in option_variants.keys()}
            with torch.inference_mode():
                out, conf, top_k_ids, top_k_probs = generate(
                    model, input_ids,
                    steps=steps, gen_length=gen_length, block_length=block_length,
                    temperature=temperature, cfg_scale=cfg_scale, remasking=remasking
                )

                # ---- Token postprocessing ----
                out = out.cpu()
                special_token_ids = set(tokenizer.all_special_ids)
                out_ids = out[:, input_ids.shape[1]:].cpu().numpy()
                conf_np = conf.detach().to(torch.float32).cpu().numpy()[:, input_ids.shape[1]:]
                valid_mask = np.isfinite(conf_np) & ~np.isin(out_ids, list(special_token_ids))
                filtered_conf = conf_np[valid_mask]

                response = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                tokens = tokenizer.convert_ids_to_tokens(out[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)
                top_k_ids = top_k_ids[:, input_ids.shape[1]:]
                top_k_probs = top_k_probs[:, input_ids.shape[1]:]

                prompt_outputs.append(response)
                per_prompt_tokens.append(tokens)

                # ---- Option probability collection ----
                batch_idx = 0
                generated_ids = out[0, input_ids.shape[1]:]
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
                for pos, token in enumerate(generated_tokens):
                    # if current token is A/B/C/D
                    for opt, variants in option_variants.items():
                        if token in variants:
                            tk_ids = top_k_ids[batch_idx, pos]
                            tk_probs = top_k_probs[batch_idx, pos]
                            tk_tokens = tokenizer.convert_ids_to_tokens(tk_ids.tolist())
                            
                            for sub_opt, sub_variants in option_variants.items():
                                for variant in sub_variants:
                                    if variant in tk_tokens:
                                        idx = tk_tokens.index(variant)
                                        prob = tk_probs[idx].item()
                                        temp_map[sub_opt] += prob
                            break
                    break
                try:
                    per_prompt_logprobs.append(float(np.log(temp_map[opt] / sum(temp_map.values()))))
                except:
                    per_prompt_logprobs.append(None)

                torch.cuda.empty_cache()

        outputs.append(prompt_outputs)
        all_tokens.append(per_prompt_tokens)
        all_logprobs.append(per_prompt_logprobs)

        del out, conf
        torch.cuda.empty_cache()

    return outputs, all_tokens, all_logprobs


def p_true_eval(model, prompts, repeat=1, steps=3, gen_length=3, block_length=3, temperature=0., cfg_scale=0., remasking='low_confidence'):

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )

    # Define True/False token variants
    tf_variants = {
        True: ["True", "true", "ĠTrue", "Ġtrue", "Yes", "yes", "ĠYes", "Ġyes"],
        False: ["False", "false", "ĠFalse", "Ġfalse", "No", "no", "ĠNo", "Ġno"]
    }

    confs = []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        m = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        temp_map = {key: 0 for key in tf_variants.keys()}

        with torch.inference_mode():
            out, conf, top_k_ids, top_k_probs = generate(
                model, input_ids,
                steps=steps, gen_length=gen_length, block_length=block_length,
                temperature=temperature, cfg_scale=cfg_scale, remasking=remasking
            )

            # ---- Token postprocessing ----
            out = out.cpu()
            special_token_ids = set(tokenizer.all_special_ids)
            out_ids = out[:, input_ids.shape[1]:].cpu().numpy()
            conf_np = conf.detach().to(torch.float32).cpu().numpy()[:, input_ids.shape[1]:]
            valid_mask = np.isfinite(conf_np) & ~np.isin(out_ids, list(special_token_ids))
            filtered_conf = conf_np[valid_mask]

            response = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            tokens = tokenizer.convert_ids_to_tokens(out[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)
            top_k_ids = top_k_ids[:, input_ids.shape[1]:]
            top_k_probs = top_k_probs[:, input_ids.shape[1]:]


            # ---- True/False probability collection ----
            batch_idx = 0
            generated_ids = out[0, input_ids.shape[1]:]
            generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
            
            for pos, token in enumerate(generated_tokens):
                for tf_key, variants in tf_variants.items():
                    if token in variants:
                        tk_ids = top_k_ids[batch_idx, pos]
                        tk_probs = top_k_probs[batch_idx, pos]
                        tk_tokens = tokenizer.convert_ids_to_tokens(tk_ids.tolist())
                        
                        for sub_tf, sub_variants in tf_variants.items():
                            for variant in sub_variants:
                                if variant in tk_tokens:
                                    idx = tk_tokens.index(variant)
                                    prob = tk_probs[idx].item()
                                    temp_map[sub_tf] += prob
                        break
                break

            try:
                p_true = temp_map[True] / sum(temp_map.values())
                confs.append(p_true)
            except:
                confs.append(None)

    return confs
