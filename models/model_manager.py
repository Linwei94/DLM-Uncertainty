from default_utils.custom_types import ModelOutputs, PromptCollection, AbstractModel
from .dream import DreamDLM
from .vllm_model import vLLMModel
from .llada import LlaDADLM
from .vllm_qwen3 import vLLMQwen3
import logging

class ModelManager:
    def __init__(self, master_cfg: dict, model_config_type="qa_model"):
        self.model_cfg: dict = master_cfg[model_config_type]
        self.model: AbstractModel
        
        match self.model_cfg.get("backend", None):
            case "vllm":
                if "qwen3" in self.model_cfg.get("name").lower():
                    # Qwen 3 has a two-stage reasoning control mechanism; hence a custom vLLM wrapper
                    logging.info("Using vLLM Qwen3 model backend")
                    self.model = vLLMQwen3(self.model_cfg)
                else:
                    logging.info("Using vLLM general model backend")
                    self.model = vLLMModel(self.model_cfg)
            case "dream":
                logging.info("Using Dream model backend")
                self.model = DreamDLM(self.model_cfg)
            case "llada":
                logging.info("Using LlaDA model backend")
                self.model = LlaDADLM(self.model_cfg)
            case "openai_batch":
                raise NotImplementedError("Not implemented yet")
            case "claude_batch":
                raise NotImplementedError("Not implemented yet")
            case "together_ai_batch":
                raise NotImplementedError("Not implemented yet")
            case None:
                raise ValueError(f"Model type not specified in config for {model_config_type}")
            case _:
                raise ValueError(f"Unknown model type: {self.model_cfg.type}")

        self.tokenizer = self.model.tokenizer


    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        return self.model.run_generation(prompt_collection)
    

    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        return self.model.run_continuation(prompt_collection)