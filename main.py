import argparse
import importlib
import pkgutil
import sys
import os
import tempfile
import multiprocessing
import logging

# Suppress transformers PyTorch 2.4 warning when using PyTorch 2.1 (vllm compatible)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Avoid tqdm/huggingface_hub conflict with vllm weight_utils.Disabledtqdm
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

from hydra import compose, initialize_config_dir
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
import pickle

from default_utils.logger import get_logger
from default_utils.utils import import_yaml_lib
from models.model_manager import ModelManager
from default_utils.datasets_manager import DatasetsManager
from default_utils.custom_types import (OrganisedOutputs, 
                                         ModelOutputs, 
                                         PromptCollection,
                                         PromptFormatterFn,
                                         OutputFilterFn,
                                         ConfidenceExtractorFn,
                                         GraderFn,
                                         MetricsFn)
from default_utils.registry import (METRICS_FUNCTIONS, 
                                     GRADER_FUNCTIONS, 
                                     CONFIDENCE_FUNCTIONS, 
                                     PROMPT_FORMATTER, 
                                     FILTER_FUNCTIONS)


def _auto_import_modules():
    import default_utils
    for package_path in default_utils.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            importlib.import_module(
                f"{default_utils.__name__}.{module_name}")
    import confidence_metrics
    for package_path in confidence_metrics.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            importlib.import_module(
                f"{confidence_metrics.__name__}.{module_name}")
    import post_processing
    for package_path in post_processing.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            importlib.import_module(
                f"{post_processing.__name__}.{module_name}")


def parse_args() -> tuple[str, str, list[str]]:
    parser = argparse.ArgumentParser(description='Run tasks with Hydra config')
    parser.add_argument('args', nargs='*',
                        help='Config overrides in key=value format')
    # Parse arguments
    args = parser.parse_args()
    # Extract task name from overrides
    dataset_name = None
    task_name = None
    other_overrides = []
    for override in args.args:
        if override.startswith('dataset='):
            dataset_name = override.split('=', 1)[1]
        elif override.startswith('task='):
            task_name = override.split('=', 1)[1]
        else:
            other_overrides.append(override)
    if not dataset_name:
        print("Error: You must specify 'dataset=<dataset_name>'")
        sys.exit(1)
    if not task_name:
        print("Error: You must specify 'task=<task_name>'")
        sys.exit(1)
    return dataset_name, task_name, other_overrides


def get_task_yaml() -> tuple[str, str, dict]:
    dataset_name, task_name, overrides = parse_args()

    # Get absolute path to config directory
    config_dir = os.path.abspath(f"./tasks/{dataset_name}")
    # Initialize Hydra with the config directory and tasks search path
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose config with the task and any overrides
        cfg = compose(
            config_name=f"{task_name}",
            overrides=overrides
        )
        return dataset_name, task_name, cfg


def _run_generation_worker(result_path: str, cfg_dict: dict, prompts: PromptCollection, generation_type: str) -> None:
    """Run generation in a subprocess. Writes (status, outputs|error) to result_path."""
    try:
        cfg = OmegaConf.create(cfg_dict)
        _auto_import_modules()
        model_manager = ModelManager(master_cfg=cfg, model_config_type="qa_model")
        if generation_type == "generation":
            outputs = model_manager.run_generation(prompts)
        else:
            outputs = model_manager.run_continuation(prompts)
        if hasattr(model_manager.model, "shutdown"):
            model_manager.model.shutdown()
        with open(result_path, "wb") as f:
            pickle.dump(("ok", outputs), f)
    except Exception as e:
        with open(result_path, "wb") as f:
            pickle.dump(("error", e), f)


def _set_third_party_log_levels() -> None:
    """Reduce noisy network/data library logs while keeping app logs at INFO."""
    noisy_loggers = [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "datasets",
        "fsspec",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # vLLM prints a warning when optional Ray deps are absent; keep it silent unless it's an error.
    ray_warning_loggers = [
        "vllm.executor.ray_utils",
        "vllm.engine.ray_utils",
        "ray_utils",
    ]
    for logger_name in ray_warning_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def main():
    # load config
    dataset_name, task_name, cfg = get_task_yaml()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # path = f"/hdd/ivny/results/{dataset_name}/{task_name}/{cfg.qa_model.name}/{timestamp}"
    path = f"results/{dataset_name}/{task_name}/{cfg.qa_model.name}/{timestamp}"
    os.makedirs(path, exist_ok=True)
    cfg["results_path"] = path
    logger = get_logger(__name__, log_file=f"{path}/task.log")
    _set_third_party_log_levels()
    _auto_import_modules()

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    logger.info(f"Preparing dataset: {dataset_name}, task: {task_name}")
    # prepare dataset
    dataset_manager: DatasetsManager = DatasetsManager(cfg)

    logger.info("Formatting prompts")
    # format prompts
    prompt_formatter: PromptFormatterFn = PROMPT_FORMATTER.get(
        cfg.get("prompt_formatter"))
    if prompt_formatter is None:
        prompt_formatter = import_yaml_lib(cfg, "prompt_formatter")
    prompts: PromptCollection = prompt_formatter(cfg, dataset_manager)

    rounds = cfg.get("rounds", 1)
    logger.info(f"Running for {rounds} rounds")

    all_rounds_eval_details = pd.DataFrame({
            'question': prompts.questions,
            'full_prompts': prompts.context_texts,
            'answer': prompts.answer_keys
    })
    all_rounds_metrics = pd.DataFrame()

    # Use spawn for multiprocessing (required for CUDA - avoids "tensor parallel already initialized" after OOM)
    ctx = multiprocessing.get_context("spawn")

    for round_idx in range(rounds):
        try:
            logger.info(f"Starting {dataset_name}, {task_name} [round {round_idx + 1}/{rounds}]")

            logger.info(f"Generating QA outputs [Round {round_idx + 1}/{rounds}]")

            # read from cache if exists to skip QA
            # --------------------------------------------------------------------
            filtered_output_path = cfg.get("filtered_output_path")
            if filtered_output_path and os.path.exists(os.path.join(filtered_output_path, f"filtered_outputs_{round_idx}.pkl")):
                logger.info("Cache for filtered_outputs found. Skipping QA generation and filtering.")
                filtered_outputs_pkl_path = os.path.join(filtered_output_path, f"filtered_outputs_{round_idx}.pkl")
                with open(filtered_outputs_pkl_path, "rb") as f:
                    outputs = pickle.load(f)
            # --------------------------------------------------------------------
            else:
                # Run generation in subprocess so OOM/parallel-init errors don't pollute subsequent rounds
                generation_type = cfg.get("generation_type", "generation")
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                    result_path = tmp.name
                try:
                    p = ctx.Process(
                        target=_run_generation_worker,
                        args=(result_path, cfg_dict, prompts, generation_type),
                    )
                    p.start()
                    p.join(timeout=3600)
                    if p.exitcode != 0 and p.exitcode is not None:
                        raise RuntimeError(
                            f"Generation subprocess exited with code {p.exitcode} "
                            "(possibly OOM - free GPU memory and retry)"
                        )
                    if not os.path.exists(result_path):
                        raise RuntimeError(
                            "Generation subprocess exited but did not write result "
                            "(possibly OOM - free GPU memory and retry)"
                        )
                    with open(result_path, "rb") as f:
                        status, data = pickle.load(f)
                    if status == "error":
                        raise data
                    outputs: list[ModelOutputs] = data
                finally:
                    if os.path.exists(result_path):
                        os.unlink(result_path)

                logger.info(f"Post-processing outputs [Round {round_idx + 1}/{rounds}]")
                # post process raw responses
                if cfg.get("output_filters") is not None:
                    for output_filter in cfg.get("output_filters", []):
                        try:
                            filter_func: OutputFilterFn = FILTER_FUNCTIONS.get(
                                output_filter.get("name"))
                            kwargs = output_filter.get("args", {})
                        except:
                            filter_func: OutputFilterFn = import_yaml_lib(
                                cfg, output_filter.get("name"))
                            kwargs = output_filter.get("args", {})
                        outputs: list[ModelOutputs] = filter_func(
                            cfg, outputs, prompts, **kwargs)

            # pickle filtered outputs for later analysis
            # --------------------------------------------------------------------
            with open(f"{path}/filtered_outputs_{round_idx}.pkl", "wb") as f:
                pickle.dump(outputs, f)
            # --------------------------------------------------------------------

            # read from cache if exists to skip QA
            # --------------------------------------------------------------------
            graded_outputs_path = cfg.get("graded_outputs_path")
            if graded_outputs_path and os.path.exists(os.path.join(graded_outputs_path, f"graded_outputs_{round_idx}.pkl")):
                logger.info("Cache for graded_outputs found. confidence extraction and grading.")
                graded_outputs_pkl_path = os.path.join(graded_outputs_path, f"graded_outputs_{round_idx}.pkl")
                with open(graded_outputs_pkl_path, "rb") as f:
                    extracted_output = pickle.load(f)
            else:
                logger.info(f"Extracting confidence scores and answers [Round {round_idx + 1}/{rounds}]")
                # extract confidence
                confidence_extraction_func: ConfidenceExtractorFn = CONFIDENCE_FUNCTIONS.get(
                    cfg.get("confidence_metrics", "length_normalised_log_likelihood"))
                if confidence_extraction_func is None:
                    confidence_extraction_func = import_yaml_lib(cfg, "confidence_metrics")
                extracted_output: OrganisedOutputs = confidence_extraction_func(
                    cfg, outputs, prompts)

                logger.info(f"Grading responses [Round {round_idx + 1}/{rounds}]")
                # grade response
                grader_func: GraderFn = GRADER_FUNCTIONS.get(
                    cfg.get("grader", "llm_grader"))
                if grader_func is None:
                    grader_func = import_yaml_lib(cfg, "grader")
                extracted_output.accuracy_scores = grader_func(
                    cfg, extracted_output, prompts, dataset_manager)

            # pickle extracted_output for later analysis
            # --------------------------------------------------------------------
            with open(f"{path}/graded_outputs_{round_idx}.pkl", "wb") as f:
                pickle.dump(extracted_output, f)
            # --------------------------------------------------------------------

            logger.info(f"Calculating performance metrics [Round {round_idx + 1}/{rounds}]")
            metrics_df = pd.DataFrame()
            # calculate metrics
            for metric in cfg.get("performance_metrics", []):
                metric_func: MetricsFn = METRICS_FUNCTIONS[metric]
                metric_value = metric_func(cfg, extracted_output)
                if isinstance(metric_value, list):
                    metrics_df[metric] = metric_value
                elif isinstance(metric_value, float):
                    metrics_df[metric] = [metric_value]
            logger.info(f"Metrics for Round {round_idx + 1}:\n{metrics_df.to_string(index=False)}")
            all_rounds_metrics = pd.concat([all_rounds_metrics, metrics_df], axis=0)

            all_rounds_eval_details = pd.concat([all_rounds_eval_details, pd.DataFrame({
                f'response_{round_idx}': extracted_output.extracted_answers[0],
                f'confidence_{round_idx}': extracted_output.extracted_confidences[0],
                f'accuracy_{round_idx}': extracted_output.accuracy_scores[0]
            })], axis=1)
        except Exception as e:
            logger.error(f"Error during round {round_idx + 1}: {e}", exc_info=True)

    # save all results
    logger.info(f"Performance Metrics:\n{all_rounds_metrics.to_string(index=False)}")
    all_rounds_metrics.to_csv(f"{path}/eval_metrics.csv", index=False)

    all_rounds_eval_details.to_csv(f"{path}/eval_details.csv", index=False)

    all_round_stats = pd.DataFrame()
    all_round_stats['metric'] = all_rounds_metrics.columns
    all_round_stats['mean'] = all_rounds_metrics.mean().values
    all_round_stats['std'] = all_rounds_metrics.std().values
    logger.info(f"Performance Summary Stats:\n{all_round_stats.to_string(index=False)}")
    all_round_stats.to_csv(f"{path}/eval_metrics_summary.csv", index=False)

    logger.info(f"All results saved to: {path}")

if __name__ == "__main__":
    main()
