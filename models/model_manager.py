from . import llada_8b, llama_3


class ModelManager():

    def __init__(self, model_name):
        self.model_id = ""
        self.model_name = model_name

    def sample(self, prompts, repeat=1, temperature=0):
        if self.model_name == "Meta-Llama-3-8B-Instruct":
            self.model_id = f"meta-llama/{self.model_name}"
            return llama_3.sample(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature)
        
        elif self.model_name == "Meta-Llama-3-8B":
            self.mnodelmodel_id_id = f"meta-llama/{self.model_name}"
            return llama_3.sample(model_id=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature)
        
        elif self.model_name == "LLaDA-8B":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.sample(model=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature)
        
        elif self.model_name == "LLaDA-8B-Instruct":
            self.model_id = f"GSAI-ML/{self.model_name}"
            return llada_8b.sample(model=self.model_id, prompts=prompts, repeat=repeat, temperature=temperature)
        
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")