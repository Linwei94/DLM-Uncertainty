import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from llada_model import LLaDAModelLM

import numpy as np

# --- 1. Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", device_map="auto")
model.eval()  # put model in eval mode

device = "cuda" if torch.cuda.is_available() else "cpu"
# model

# --- 2. Prepare input ---
prompt = "What is the capital of Australia?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids  # shape (1, seq_len)

# --- 3. Generate tokens step-by-step to track log-probs ---
max_new_tokens = 2
generated_ids = input_ids
past_key_values = None
all_log_probs = []

for _ in range(max_new_tokens):
    # Prepare inputs for generation
    model_inputs = model.prepare_inputs_for_generation(
        generated_ids, past_key_values=past_key_values
    )
    
    # Forward pass
    outputs = model(**model_inputs)
    logits = outputs.logits[:, -1, :]  # only last token logits
    past_key_values = outputs.past_key_values  # cache for next step
    
    # Convert to log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Greedy sample: pick the token with highest probability
    next_token_id = torch.argmax(log_probs, dim=-1, keepdim=True)
    
    # Store log-prob of the chosen token
    token_log_prob = log_probs.gather(-1, next_token_id).item()
    all_log_probs.append(token_log_prob)
    
    # Append to sequence
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

# --- 4. Decode the generated sequence ---
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Generated text:\n", generated_text)
print("Token log-probabilities:\n", all_log_probs)
print("Confidence:\n", np.exp(np.mean(all_log_probs)))
