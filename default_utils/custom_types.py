from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Callable
from .datasets_manager import DatasetsManager

@dataclass
class PromptCollection:
    questions: list[str] = field(default_factory=list)  # a list of questions
    answer_keys: list[str] = field(default_factory=list)  # a list of answer keys
    system_prompt: str = ""  # optional system prompt
    context_texts: list[str] = field(default_factory=list) # a list of strings
    continuation_texts : dict[str, list[str]] = field(default_factory=dict) # context (str) and continuations (list of strings) mapping


@dataclass
class ModelOutputs: # each object corresponds to one sampling (generation) round
    context_texts: list[str] = field(default_factory=list) # a list of strings
    output_texts: list[str] = field(default_factory=list) # a list of strings
    output_tokens: list[list[str]] = field(default_factory=list) # a list of strings
    output_logprobs: list[list[float]] = field(default_factory=list) # a list of floats
    continuation_candidates: list[list[dict]] = field(default_factory=list) 
    top_k_tokens: list[list[list[tuple[str, float]]]] = field(default_factory=list) 


@dataclass
class OrganisedOutputs:
    """
    Each sublist corresponds to a sampling round
    """
    extracted_answers: list[list[str]] = field(default_factory=list) 
    extracted_confidences: list[list[float]] | list[list[object]] = field(default_factory=list)
    accuracy_scores: list[list[float]] | list[list[object]] = field(default_factory=list)


class AbstractModel(ABC):
    tokenizer: object = None
    
    @abstractmethod
    def run_generation(self, prompts: PromptCollection) -> "ModelOutputs":
        """Generate outputs given prompts (e.g., free-form generation)."""
        raise NotImplementedError

    @abstractmethod
    def run_continuation(self, prompts: PromptCollection) -> "ModelOutputs":
        """Score or generate continuations conditioned on contexts."""
        raise NotImplementedError


PromptFormatterFn = Callable[[dict, DatasetsManager], PromptCollection]
OutputFilterFn = Callable[[dict, list[ModelOutputs], PromptCollection], list[ModelOutputs]]
ConfidenceExtractorFn = Callable[[dict, list[ModelOutputs], PromptCollection], OrganisedOutputs]
GraderFn = Callable[[dict, OrganisedOutputs, PromptCollection, DatasetsManager], list[list[float]]]
MetricsFn = Callable[[dict, OrganisedOutputs], list[float]]