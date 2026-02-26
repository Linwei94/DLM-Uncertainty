import pandas as pd
from default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from default_utils.datasets_manager import DatasetsManager
import numpy as np

def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["question"] = dataset["question"].str.capitalize()
    dataset["answer"] = dataset["correct_answers"].values
    print(dataset)
    return dataset

