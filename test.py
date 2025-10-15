from pre_processing.dataset_fetcher import DatasetFetcher
from pre_processing.prompt_fomatter import PromptFormatter


df = DatasetFetcher("mmlu_mini").dataset
print(PromptFormatter(df, "mmlu_5").prompts[0])