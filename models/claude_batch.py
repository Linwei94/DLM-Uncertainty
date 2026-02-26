from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection

class ClaudeBatch(AbstractModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        pass
        
    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        pass