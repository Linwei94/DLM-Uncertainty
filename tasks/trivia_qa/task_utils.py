import pandas as pd
from default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from default_utils.datasets_manager import DatasetsManager
import numpy as np


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the TruthfulQA dataset according to the configuration.
    """
    dataset["answer"] = dataset["answer"].apply(lambda x: x.get("aliases", []))
    return dataset