import pandas as pd
from default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from default_utils.datasets_manager import DatasetsManager
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["answer"] = dataset["answers"].apply(lambda x: x['text'][0] if isinstance(x, dict) and 'text' in x and len(x['text']) > 0 else "Unanswerable")
    return dataset

