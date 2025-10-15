from vllm import LLM, SamplingParams
from tqdm import tqdm  # optional progress bar

def sample(model_id, prompts, repeat=1, max_tokens=5, temperature=0, show_progress=True):
    """
    Samples text from an LLM multiple times and groups outputs by prompt.
    
    Args:
        model_id (str): Model name or path (e.g. 'meta-llama/Meta-Llama-3-8B').
        prompts (List[str]): List of prompt strings.
        repeat (int): Number of repetitions per prompt.
        max_tokens (int): Max tokens per generation.
        show_progress (bool): Show tqdm progress bar.
    
    Returns:
        all_responses (List[List[str]]): responses[prompt_idx][round_idx]
        all_tokens (List[List[List[str]]]): tokens[prompt_idx][round_idx][token_idx]
        all_logprobs (List[List[List[float]]]): logprobs[prompt_idx][round_idx][token_idx]
    """
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
