from ..default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection

class LlaDADLM(AbstractModel):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def run_generation(self, prompt_collection: PromptCollection) -> ModelOutputs:
        # Implement generation logic here
        pass
        
    def run_continuation(self, prompt_collection: PromptCollection) -> ModelOutputs:
        # Implement conditional likelihood calculation here
        pass