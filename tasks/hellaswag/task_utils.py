import pandas as pd
from default_utils.custom_types import ModelOutputs, PromptCollection
from default_utils.datasets_manager import DatasetsManager
from jinja2 import Template
import re

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset according to the configuration.
    """
    # Example preprocessing: drop rows with missing values
    dataset["complete_ctx"] = dataset["ctx_a"] + " " + dataset["ctx_b"].str.capitalize()
    dataset["question"] = (dataset["activity_label"] + ": " + dataset["complete_ctx"]).apply(preprocess)
    dataset["answer_index"] = dataset["label"].astype(int).values
    dataset["choices"] = dataset["endings"].apply(lambda endings: [preprocess(ending) for ending in endings])
    return dataset


def hella_swag_prompt_formatter(cfg: dict, dataset_manager: DatasetsManager) -> PromptCollection:
    question_template: Template = Template(cfg.get("question_format", """{{ question.strip() }}\n{% for choice in choices %} {{ 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[loop.index0] }}. {{ choice }}\n{% endfor %}"""))
    prompt_template: Template = Template(cfg.get("prompt_format", "{{few_shot_examples}}\n\n{{formatted_question}}\nAnswer:"))
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
                few_shot_text += question_template.render(**example.to_dict()).strip() + "\n"
                few_shot_text += f"Answer: {example['choices'][example['answer_index']]}\n\n"
        formatted_question = question_template.render(**row.to_dict()).strip()
        formatted_questions.append(formatted_question)
        base_prompt = prompt_template.render(few_shot_examples=few_shot_text.strip(), formatted_question=formatted_question).strip()
        continuations = []
        for i in row['choices']:
            continuations.append(i)
        prompts.append(base_prompt)
        continuation_texts[base_prompt] = continuations
    
    print(formatted_questions[0])
    return PromptCollection(questions=formatted_questions, answer_keys=[choices[answer_index] for choices, answer_index in zip(ds["choices"], ds["answer_index"])], system_prompt=cfg.get("system_prompt", ""), context_texts=prompts, continuation_texts=continuation_texts)