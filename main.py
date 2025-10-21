import argparse
import pandas as pd
import os
import ast
import time

from pre_processing.dataset_fetcher import DatasetFetcher
from pre_processing.prompt_fomatter import PromptFormatter

from post_processing.confidence_estimator import ConfidenceEstimator
from post_processing.grader import Grader
from post_processing.metrics import Metrics

from models.model_manager import ModelManager

def parse_args():
    parser = argparse.ArgumentParser(description="Example script using dataset, prompt, and conf arguments")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the dataset to use"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path or name of the dataset to use"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt or instruction text"
    )

    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Confidence method"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = args.model
    dataset_name = args.dataset
    prompt = args.prompt
    conf = args.conf

    dataset: pd.DataFrame = DatasetFetcher(dataset_name).dataset.copy()
    prompts = PromptFormatter(dataset, prompt_type=prompt).prompts

    repeat_map = {
        "semantic_uncertainty": 10,
    }

    temperature_map = {
        "semantic_uncertainty": 1,
    }

    repeat = repeat_map.get(conf, 1)  # repeat 10 times for sentence uncertainty
    temperature = temperature_map.get(conf, 0)

    raw_response_path = f"cache/{dataset_name}/{model}/{conf}/{prompt}_raw_responses.csv"
    extracted_response_path = f"cache/{dataset_name}/{model}/{conf}/{prompt}_extracted_responses.csv"
    results_response_path = f"results/{dataset_name}/{model}/{conf}/{prompt}_graded_responses.csv"

    os.makedirs(f"cache/{dataset_name}/{model}/{conf}", exist_ok=True)
    os.makedirs(f"results/{dataset_name}/{model}/{conf}", exist_ok=True)

    qa_model = ModelManager(model, dataset_name)

    # sampling
    if os.path.exists(raw_response_path):
        dataset = pd.read_csv(raw_response_path, converters={
            "raw_response": ast.literal_eval,
            "tokens": ast.literal_eval,
            "logprobs": ast.literal_eval
        })
        print("Skipping sampling, raw responses already exist.")
    else: 
        time.sleep(10)
        if "vnc" in conf.lower():
            raw_responses, all_tokens, all_logprobs = qa_model.sample(prompts=prompts, repeat=repeat, temperature=temperature)
        else:
            sampling_methods = {
                "mmlu": qa_model.sample_4_choices,
                "mmlu_mini": qa_model.sample_4_choices,
                "mmlu_pro": None,
                "mmlu_pro_mini": None,
                "truthful_qa": qa_model.sample,
                "truthful_qa_mini": qa_model.sample,
                "gpqa": qa_model.sample_4_choices,
                "gpqa_mini": qa_model.sample_4_choices,
                "hellaswag": None,
                "hellaswag_mini": None,
                "arcc": None,
                "arcc_mini": None,
                "gsm8k": None,
                "gsm8k_mini": None,
            }
            raw_responses, all_tokens, all_logprobs = sampling_methods[dataset_name](prompts=prompts, repeat=repeat, temperature=temperature)
            
        dataset["raw_response"] = raw_responses
        dataset["tokens"] = all_tokens
        dataset["logprobs"] = all_logprobs
        dataset.to_csv(raw_response_path, index=False)
    

    # confidence
    if os.path.exists(extracted_response_path):
        dataset = pd.read_csv(extracted_response_path, converters={
            "raw_response": ast.literal_eval,
            "tokens": ast.literal_eval,
            "logprobs": ast.literal_eval
        })
        print("Skipping confidence extraction.")
    else: 
        import torch
        torch.cuda.empty_cache()
        time.sleep(10)
        estimator = ConfidenceEstimator(dataset_name=dataset_name, dataset=dataset, confidence_type=conf, self_eval_model=qa_model)
        dataset["response"] = estimator.responses
        dataset["confidence"] = estimator.confidence_scores
        dataset.to_csv(extracted_response_path, index=False)


    # grading
    if os.path.exists(results_response_path):
        dataset = pd.read_csv(results_response_path, converters={
            "raw_response": ast.literal_eval,
            "tokens": ast.literal_eval,
            "logprobs": ast.literal_eval
        })
        print("Skipping grading.")
    else:
        import torch
        torch.cuda.empty_cache()
        time.sleep(10)
        grader = Grader(dataset=dataset, dataset_name=dataset_name)
        dataset["grade"] = grader.grades
        dataset.to_csv(results_response_path, index=False)
        

    # results
    metrics = Metrics(grades=dataset["grade"].tolist(), confidence_scores=dataset["confidence"].tolist())
    print("Evaluation Complete:", model, dataset_name, conf, prompt)
    print(metrics.save(f"results/{dataset_name}/{model}/{conf}/{prompt}_summary.csv"))