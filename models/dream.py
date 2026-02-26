import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection


class DreamDLM(AbstractModel):
    """
    Dream DLM model wrapper.

    References:
    - https://github.com/DreamLM/Dream/tree/main
    - https://arxiv.org/abs/2508.15487
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left') 
    
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = self.tokenizer
        model_outputs_list = []
        for _ in range(self.repeat):
            messages_list = []
            for context_text in prompt_collection.context_texts:
                messages = [{"role": "user", "content": context_text}]
                messages_list.append(messages)
            output_texts = []
            output_tokens = []
            for msg in tqdm(messages_list, desc="Generating outputs"):
                inputs = tokenizer.apply_chat_template(msg, 
                                                    return_tensors="pt", 
                                                    return_dict=True, 
                                                    add_generation_prompt=True, 
                                                    padding=True)
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.cfg.get("max_tokens", 256),
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=self.cfg.get("max_tokens", 256),
                    temperature=self.cfg.get("temperature", 1.0),
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                )

                generations = [
                    tokenizer.decode(g[len(p) :].tolist())
                    for p, g in zip(input_ids, output.sequences)
                ]
                output_texts.append(generations[0].split(tokenizer.eos_token)[0])
                output_tokens.append([tokenizer.decode(t) for t in tokenizer.encode(output_texts[-1])])

            # obtain logprobs
            logprob_prompt_collection = prompt_collection
            logprob_prompt_collection.continuation_texts = {
                context: [text] for context, text in zip(prompt_collection.context_texts, output_texts)
            }
            cont_outputs = self.run_continuation(logprob_prompt_collection)[0]

            model_outputs_list.append(
                ModelOutputs(
                    context_texts=prompt_collection.context_texts,
                    output_texts=output_texts,
                    output_tokens=output_tokens,
                    output_logprobs=cont_outputs.output_logprobs,
                )
            )

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return model_outputs_list

    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:

        hf_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,     # model weights can be BF16
            device_map="auto",
            trust_remote_code=True,
        )

        model_outputs_list = []
        
        for _ in range(self.repeat):
            all_output_texts = []
            all_output_tokens = []
            all_output_logprobs = []
            all_candidates = []

            for context in tqdm(prompt_collection.context_texts, desc="Scoring continuations"):
                continuations = prompt_collection.continuation_texts[context]

                # Store all continuations with their scores
                candidates = []

                for continuation in continuations:

                    # 1. prepare full prompt
                    full_prompt = context + " " + continuation

                    # 2. tokenize (keep as Python ints)
                    ctx_ids  = self.tokenizer(context, add_special_tokens=False).input_ids
                    full_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
                    cont_ids = full_ids[len(ctx_ids):]

                    # 3. convert to tensor (must be int64)
                    input_ids = torch.tensor([full_ids], dtype=torch.long).to(hf_model.device)

                    # 4. forward pass
                    with torch.no_grad():
                        outputs = hf_model(input_ids=input_ids)
                        logits = outputs.logits  # [1, seq, vocab]

                    # 5. logprobs
                    logprobs = F.log_softmax(logits, dim=-1)

                    # 6. extract continuation logprobs
                    cont_tokens = []
                    cont_logps = []

                    offset = len(ctx_ids)

                    for i, token_id in enumerate(cont_ids):

                        tok_str = self.tokenizer.decode([token_id])

                        # logprob from the previous position
                        logp = logprobs[0, offset + i - 1, token_id].item()

                        cont_tokens.append(tok_str)
                        cont_logps.append(logp)

                    # Calculate sum of logprobs for this continuation
                    logprob_mean = np.mean(cont_logps)
                    
                    candidates.append({
                        'text': continuation,
                        'tokens': cont_tokens,
                        'logprobs': cont_logps,
                        'mean': logprob_mean
                    })

                # Select the continuation with maximum sum of logprobs
                best_candidate = max(candidates, key=lambda x: x['mean'])
                all_candidates.append(candidates)
                # 7. store only the best continuation
                all_output_texts.append(best_candidate['text'])
                all_output_tokens.append(best_candidate['tokens'])
                all_output_logprobs.append(best_candidate['logprobs'])

            model_outputs_list.append(
                ModelOutputs(
                    context_texts=prompt_collection.context_texts,
                    output_texts=all_output_texts,
                    output_tokens=all_output_tokens,
                    output_logprobs=all_output_logprobs,
                    continuation_candidates=all_candidates
                )
            )

        del hf_model
        gc.collect()
        torch.cuda.empty_cache()
        return model_outputs_list
