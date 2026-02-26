import pandas as pd
from ...default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from ...default_utils.datasets_manager import DatasetsManager
import numpy as np
from jinja2 import Template


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the TruthfulQA dataset according to the configuration.
    """
    # extract "#### (\\-?[0-9\\.\\,]+)" from answer
    dataset["original_answer"] = dataset["answer"].values
    dataset["answer"] = dataset["answer"].str.extract(r"#### (\-?[0-9\.\,]+)")[0]
    print(dataset)
    return dataset


def gsm8k_prompt_formatter(cfg: dict, dataset_manager: DatasetsManager) -> PromptCollection:
    question_template: Template = Template(cfg.get("question_format", "{{question.strip()}}"))
    prompt_template: Template = Template(cfg.get("prompt_format", "{{few_shot_examples}}\n\nQ: {{formatted_question}}\nA:"))
    prompts = []
    few_shot = dataset_manager.few_shot_examples if dataset_manager.few_shot_examples else 0
    # Iterate through the dataset
    continuation_texts = dict()
    formatted_questions = []
    ds: pd.DataFrame = dataset_manager.get_dataset()
    few_shot_ds = dataset_manager.get_few_shot_dataset()
    for _, row in ds.iterrows():
        # Build few-shot examples if requested
        few_shot_text = ""
        if few_shot > 0:
            few_shot_examples: pd.DataFrame = few_shot_ds.sample(n=few_shot, random_state=cfg.get("seed", 42))
            for _, example in few_shot_examples.iterrows():
                few_shot_text += "Question: " + question_template.render(**example.to_dict()).strip() + "\n"
                few_shot_text += f"Answer: {example['original_answer']}\n\n"
        formatted_question = question_template.render(**row.to_dict()).strip()
        formatted_questions.append(formatted_question)
        base_prompt = prompt_template.render(few_shot_examples=few_shot_text.strip(), formatted_question=formatted_question).strip()
        prompts.append(base_prompt)
    return PromptCollection(questions=formatted_questions, answer_keys=ds['answer'].tolist(), system_prompt=cfg.get("system_prompt", ""), context_texts=prompts, continuation_texts=continuation_texts)