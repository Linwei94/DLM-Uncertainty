import os
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os 
import shutil

# source path (your modified file)
src = "/home/ivan/DLM-Uncertainty/models/custom_dream_generation_utils.py"
# destination path (cached Hugging Face location)
dst_dir = "/hdd/.cache/huggingface/modules/transformers_modules/Dream-org/Dream-v0-Instruct-7B/05334cb9faaf763692dcf9d8737c642be2b2a6ae"
dst = os.path.join(dst_dir, "generation_utils.py")
# make sure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)
# copy the file
shutil.copy2(src, dst)

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [[
    {"role": "user", "content": "Write a story that ends with 'Finally, Joey and Rachel get married.'"}
],
[
    {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}
]]
# set padding=True
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True, padding=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    output_history=True,
    return_dict_in_generate=True,
    steps=256,
    temperature=0.,
    top_p=0.95,
    alg="origin",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output["sequences"])
]

print(generations[0].split(tokenizer.eos_token)[0])
print("=="*20)
print(generations[1].split(tokenizer.eos_token)[0])

print(output["confidence"])