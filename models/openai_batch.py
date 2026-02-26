from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from openai import OpenAI
import json
import os
import time
import logging


class OpenAIBatch(AbstractModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        batch_ids = self.cfg.get("cache", None)

        # If cache (batch_ids) provided, retrieve results
        if batch_ids:
            if isinstance(batch_ids, str):
                batch_ids = [batch_ids]

            model_outputs_list = []
            for batch_id in batch_ids:
                responses = self._retrieve_batch_job_output(batch_id)

                # Extract tokens and logprobs
                output_texts = []
                output_tokens = []
                output_logprobs = []

                for response_data in responses:
                    output_texts.append(response_data["text"])
                    output_tokens.append(response_data["tokens"])
                    output_logprobs.append(response_data["logprobs"])

                model_outputs_list.append(ModelOutputs(
                    context_texts=prompt_collection.context_texts,
                    output_texts=output_texts,
                    output_tokens=output_tokens,
                    output_logprobs=output_logprobs,
                ))
            return model_outputs_list

        # Otherwise, create and submit batch jobs
        batch_ids = []
        for repeat_idx in range(self.repeat):
            batch_id = self._prepare_batch_task_and_submit(
                prompt_collection,
                task_name=f"generation_repeat_{repeat_idx}_{self.model_name}"
            )
            batch_ids.append(batch_id)
            logging.info(
                f"Submitted batch job {batch_id} for repeat {repeat_idx} with model {self.model_name}")

        # Wait for all batches to complete
        model_outputs_list = []
        for batch_id in batch_ids:
            while True:
                batch_job = self.client.batches.retrieve(batch_id)
                if batch_job.status != "completed":
                    logging.info(
                        f"Batch job {batch_id} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_id} is completed")
                    break

            responses = self._retrieve_batch_job_output(batch_id)

            # Extract tokens and logprobs from responses
            output_texts = []
            output_tokens = []
            output_logprobs = []

            for response_data in responses:
                output_texts.append(response_data["text"])
                output_tokens.append(response_data["tokens"])
                output_logprobs.append(response_data["logprobs"])

            model_outputs_list.append(ModelOutputs(
                context_texts=prompt_collection.context_texts,
                output_texts=output_texts,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
            ))

        logging.info(f"All batch jobs completed. Batch IDs: {batch_ids}")
        return model_outputs_list

    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        raise NotImplementedError(
            "Continuation not implemented for OpenAIBatch model.")

    def _prepare_batch_task_and_submit(self, prompt_collection: PromptCollection, task_name: str) -> str:
        """Prepare and submit a batch job for generation"""
        tasks = []
        for idx, context_text in enumerate(prompt_collection.context_texts):
            messages = []
            if prompt_collection.system_prompt:
                messages.append(
                    {"role": "system", "content": prompt_collection.system_prompt})
            else:
                messages.append(
                    {"role": "system", "content": "You are a helpful assistant."})
            messages.append({"role": "user", "content": context_text})

            task = {
                "custom_id": f"{task_name}_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "temperature": self.cfg.get("temperature", 1.0),
                    "max_tokens": self.cfg.get("max_tokens", 256),
                    "logprobs": True,
                    "top_logprobs": 1,
                    "messages": messages
                }
            }

            # Add reasoning_effort if specified in config
            if self.cfg.get("reasoning_effort"):
                task["body"]["reasoning_effort"] = self.cfg.get(
                    "reasoning_effort")

            tasks.append(task)

        # Write tasks to temporary JSONL file
        task_file_path = f"/tmp/openai_batch_{task_name}.jsonl"
        with open(task_file_path, 'w', encoding='utf-8') as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

        # Upload and create batch job
        batch_file = self.client.files.create(
            file=open(task_file_path, 'rb'),
            purpose='batch'
        )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        return batch_job.id

    def _retrieve_batch_job_output(self, batch_job_id: str) -> list[dict]:
        """Retrieve outputs from a completed batch job with tokens and logprobs"""
        batch_job = self.client.batches.retrieve(batch_job_id)

        # Parse results file
        result_bytes = self.client.files.content(
            batch_job.output_file_id).content
        result_text = result_bytes.decode("utf-8")

        merged = {}
        for line in result_text.strip().split("\n"):
            entry = json.loads(line)
            logging.debug(f"Batch job entry: {entry}")
            custom_id = entry["custom_id"]

            choice = entry["response"]["body"]["choices"][0]
            text = choice["message"]["content"].strip()

            # Extract tokens and logprobs
            tokens = []
            logprobs = []

            if "logprobs" in choice and choice["logprobs"] and choice["logprobs"]["content"]:
                for token_data in choice["logprobs"]["content"]:
                    tokens.append(token_data["token"])
                    logprobs.append(token_data["logprob"])

            merged[custom_id] = {
                "text": text,
                "tokens": tokens,
                "logprobs": logprobs
            }

        # Parse errors file if present
        if batch_job.error_file_id:
            error_bytes = self.client.files.content(
                batch_job.error_file_id).content
            error_text = error_bytes.decode("utf-8")

            for line in error_text.strip().split("\n"):
                entry = json.loads(line)
                custom_id = entry["custom_id"]
                if custom_id not in merged:
                    merged[custom_id] = {
                        "text": "",
                        "tokens": [],
                        "logprobs": []
                    }

        # Sort responses by trailing number in custom_id
        responses = [
            merged[k]
            for k in sorted(
                merged.keys(),
                key=lambda x: int(x.rsplit("_", 1)[-1])
            )
        ]

        return responses
