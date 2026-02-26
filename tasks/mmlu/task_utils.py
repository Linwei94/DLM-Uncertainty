import pandas as pd
from default_utils.custom_types import ModelOutputs

def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset according to the configuration.
    """
    # Example preprocessing: drop rows with missing values
    dataset["answer_index"] = dataset["answer"].values
    return dataset