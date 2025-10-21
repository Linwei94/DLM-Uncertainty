from . import llada_8b, llama_3
from configs import *

class ModelManager():

    def __init__(self, model_name, dataset_name):
        self.model_id = ""
        self.model_name = model_name
        self.llada_gen_length = LLADA_GEN_LENGTH_MAP[dataset_name]
        self.hf_gen_length = AR_GEN_LENGTH_MAP[dataset_name]

    def sample(self, prompts, repeat=1, temperature=0):
        if self.model_name == "Meta-Llama-3-8B-Instruct":
            self.model_id = f"meta-llama/{self.model_name}"
            return llama_3.sample(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, max_tokens=self.hf_gen_length)
        
        elif self.model_name == "Meta-Llama-3-8B":
            self.mnodelmodel_id_id = f"meta-llama/{self.model_name}"
            return llama_3.sample(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, max_tokens=self.hf_gen_length)
        
        elif self.model_name == "LLaDA-8B":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.sample(model=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, gen_length=self.llada_gen_length, block_length=self.llada_gen_length, steps=self.llada_gen_length)
        
        elif self.model_name == "LLaDA-8B-Instruct":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.sample(model=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, gen_length=self.llada_gen_length, block_length=self.llada_gen_length, steps=self.llada_gen_length)
        
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")
        

    def sample_4_choices(self, prompts, repeat=1, temperature=0):
        if self.model_name == "Meta-Llama-3-8B-Instruct":
            self.model_id = f"meta-llama/{self.model_name}"
            return llama_3.sample_4_choices(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, max_tokens=self.hf_gen_length)
        
        elif self.model_name == "Meta-Llama-3-8B":
            self.mnodelmodel_id_id = f"meta-llama/{self.model_name}"
            return llama_3.sample_4_choices(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, max_tokens=self.hf_gen_length)
        
        elif self.model_name == "LLaDA-8B":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.sample_4_choices(model=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, gen_length=self.llada_gen_length, block_length=self.llada_gen_length, steps=self.llada_gen_length)
        
        elif self.model_name == "LLaDA-8B-Instruct":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.sample_4_choices(model=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature, gen_length=self.llada_gen_length, block_length=self.llada_gen_length, steps=self.llada_gen_length)
        
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")
        
    
    def p_true_eval(self, prompts, repeat=1, temperature=0):
        if self.model_name == "Meta-Llama-3-8B-Instruct":
            self.model_id = f"meta-llama/{self.model_name}"
            return llama_3.p_true_eval(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature)
        
        elif self.model_name == "Meta-Llama-3-8B":
            self.mnodelmodel_id_id = f"meta-llama/{self.model_name}"
            return llama_3.p_true_eval(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature)
        
        elif self.model_name == "LLaDA-8B":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.p_true_eval(model=self.model_id, prompts=prompts, temperature=temperature)
        
        elif self.model_name == "LLaDA-8B-Instruct":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.p_true_eval(model=self.model_id, prompts=prompts, temperature=temperature)
        
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")