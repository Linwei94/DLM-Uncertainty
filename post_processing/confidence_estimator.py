import pandas as pd
import re 
import json
import numpy as np 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.model_manager import ModelManager

class ConfidenceEstimator:
    
    def __init__(self, dataset_name: str, dataset: pd.DataFrame, confidence_type: str = "vnc", self_eval_model: ModelManager = None):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.raw_responses = dataset["raw_response"].tolist()
        self.confidence_type = confidence_type
        self.responses: list[str] = dataset["raw_response"].tolist()
        self.confidence_scores: list[float] = []
        self.self_eval_model: ModelManager = self_eval_model
        self.estimate_confidence()


    def estimate_confidence(self):
        if self.confidence_type == "vnc":
            self.vnc_estimator()
        elif self.confidence_type == "semantic_uncertainty":
            self.semantic_uncertainty_estimator()
        elif self.confidence_type == "token_probs":
            self.token_probs_estimator()
        elif self.confidence_type == "linguistic_confidecne":
            self.linguistic_confidence()
        elif self.confidence_type == "p_true":
            self.p_true_token_probs()
        else:
            raise NotImplementedError(f"Confidence type {self.confidence_type} not implemented.")

    
    def semantic_uncertainty_estimator(self) -> tuple[list[float], list[str]]:
        """
        Args:
            response_lists: List of list of responses for each question.
                            Each inner list contains repeated samples for the same question.
            entailment_model: An instance of EntailmentDeberta (or similar) with check_implication_batch method.
            strict_entailment: Whether to require strict bidirectional entailment for equivalence.

        Returns:
            confidences: List of float confidence scores per question.
            selected_responses: List of selected responses per question.
        """
        # if "mmlu" in self.dataset_name:
        #     response_lists = [x[0].split("\n")[0][-10:] if len(x[0].split("\n")) > 1 else x[0] for x in self.raw_responses]
        # else:
        response_lists = self.raw_responses
        entailment_model = EntailmentDeberta()
        strict_entailment: bool = False

        confidences = []
        selected_responses = []

        for response_set in tqdm(response_lists, desc="Processing questions"):
            # Step 1: Compute semantic IDs
            n = len(response_set)
            if n == 1:
                semantic_ids = [0]
            else:
                # Build all unique pairs
                left = []
                right = []
                for i in range(n):
                    for j in range(i+1, n):
                        left.append(response_set[i])
                        right.append(response_set[j])
                
                # Check semantic equivalence
                batch_results = entailment_model.check_implication_batch(left, right)
                
                # Map results to boolean equivalence
                def are_equivalent(idx1, idx2):
                    pair_idx = idx1 * (n - 1) - (idx1 * (idx1 + 1)) // 2 + (idx2 - idx1 - 1)
                    i1 = batch_results[pair_idx]
                    i2 = batch_results[pair_idx]  # symmetric
                    if strict_entailment:
                        return i1 == 2 and i2 == 2
                    else:
                        return i1 != 0 and i2 != 0 and not (i1 == 1 and i2 == 1)

                # Assign semantic IDs
                semantic_ids = [-1] * n
                next_id = 0
                for i in range(n):
                    if semantic_ids[i] == -1:
                        semantic_ids[i] = next_id
                        for j in range(i + 1, n):
                            if are_equivalent(i, j):
                                semantic_ids[j] = next_id
                        next_id += 1

            # Step 2: Find most frequent semantic ID
            most_freq_id = max(set(semantic_ids), key=semantic_ids.count)
            confidence = semantic_ids.count(most_freq_id) / len(semantic_ids)
            confidences.append(confidence)

            # Step 3: Pick the first response with that semantic ID
            selected_response = response_set[semantic_ids.index(most_freq_id)]
            selected_responses.append(selected_response)

        self.confidence_scores = confidences
        self.responses = selected_responses


    def token_probs_estimator(self):
        self.confidence_scores = []
        for x in self.dataset["logprobs"].values:
            try:
                self.confidence_scores.append(float(np.exp(np.nanmean(x))))
            except:
                self.confidence_scores.append(np.nan)

    
    def linguistic_confidence(self):
        raise NotImplementedError(f"Confidence type {self.confidence_type} not implemented.")


    def vnc_estimator(self):
        # Load the model once
        model_id = "openai/gpt-oss-20b"
        llm = LLM(model=model_id, dtype="bfloat16", max_model_len=4096)

        # Sampling configuration
        sampling_params = SamplingParams(
            max_tokens=256
        )

        prompts = []
        for text in self.raw_responses:
            if "mmlu" in self.dataset_name:
                prompt = f"""
                You are a strict information extractor. The text is an answer followed by a confidence socre. Extract ONLY what is contained in the given text,
                without using any outside knowledge. Return a valid JSON object strictly in this format:

                {{
                    "answer": "<the answer extracted from the text, or null if none. It is a single letter if it is a multiple choice question.>",
                    "confidence_score": <a number between 0 and 100 estimating confidence, verbalised in the text. None if not present.>
                }}

                Text:
                {text}
                """
            else:
                prompt = f"""
                You are a strict information extractor. The text is an answer followed by a confidence socre. Extract ONLY what is contained in the given text,
                without using any outside knowledge. Return a valid JSON object strictly in this format:
                {{
                    "answer": "<the answer extracted from the text, or null if none.>",
                    "confidence_score": <a number between 0 and 100 estimating confidence, verbalised in the text. None if not present.>
                }}

                Text:
                {text}
                """
            prompts.append([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt.strip()}])

        # Run generation
        outputs = llm.chat(prompts, sampling_params, chat_template_kwargs={"reasoning_effort": "low"})

        cleaned = []
        for output in outputs:
            text = output.outputs[0].text.strip().rsplit("assistantfinal", 1)[-1]
            # Try to isolate JSON substring
            start = text.find("{")
            end = text.rfind("}")
            try:
                parsed = json.loads(text[start:end+1])
                cleaned.append(parsed)
            except Exception:
                cleaned.append({})
        # Extract structured fields
            
        self.responses = [
            x.get("answer") if isinstance(x, dict) else None
            for x in cleaned
        ]
        self.confidence_scores = np.array([
            (
                float(x.get("confidence_score")) / 100.0
                if isinstance(x, dict)
                and x.get("confidence_score") is not None
                and str(x.get("confidence_score")).replace('.', '', 1).isdigit()
                else np.nan
            )
            for x in cleaned
        ])


    def p_true_monte_carlo(self):
        # preprare self eval prompt
        P_TRUE_SELF_EVALUATION_PROMPT = """
        Question: {question}
        Proposed Answer: {proposed_answer}
        Is the proposed answer:
         True
         False
        Output either True or False with no other text around it.
        """.strip()

        # if "mmlu" in self.dataset_name:
        #     raw_responses = [x[0].split("\n")[0][-10:] if len(x[0].split("\n")) > 1 else x[0] for x in self.raw_responses]
        # else:
        raw_responses = [x[0] for x in self.raw_responses]
        prompts = [P_TRUE_SELF_EVALUATION_PROMPT.format(question=q, proposed_answer=r) for q,r in zip(self.dataset["question"], raw_responses)]
        eval_list, _, _ = self.self_eval_model.sample(prompts=prompts, repeat=10, temperature=1)
        confidences = []
        for self_evals in eval_list:
            true_count = sum("true" in str(x).lower() for x in self_evals)
            false_count = sum("false" in str(x).lower() for x in self_evals)
            if true_count + false_count > 0:
                confidences.append(true_count / (true_count + false_count))
            else:
                confidences.append(float("nan"))
        self.confidence_scores = confidences
        self.responses = raw_responses


    def p_true_token_probs(self):
        # preprare self eval prompt
        P_TRUE_SELF_EVALUATION_PROMPT = """
        Question: {question}
        Proposed Answer: {proposed_answer}
        Is the proposed answer:
         True
         False
        Output either True or False with no other text around it.
        """.strip()

        # if "mmlu" in self.dataset_name:
        #     raw_responses = [x[0].split("\n")[0][-10:] if len(x[0].split("\n")) > 1 else x[0] for x in self.raw_responses]
        # else:
        raw_responses = [x[0] for x in self.raw_responses]
        prompts = [P_TRUE_SELF_EVALUATION_PROMPT.format(question=q, proposed_answer=r) for q,r in zip(self.dataset["question"], raw_responses)]
        confidences = self.self_eval_model.p_true_eval(prompts=prompts, temperature=0)
        self.confidence_scores = confidences
        self.responses = raw_responses


class EntailmentDeberta():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(device)

    def check_implication(self, text1, text2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        return prediction

    def check_implication_batch(
        self,
        texts1,
        texts2,
        batch_size: int = 256,
        max_length: int = 256,
    ):
        """
        Batched inference for pairs (premise -> hypothesis).

        Args:
            texts1: List[str] of premises.
            texts2: List[str] of hypotheses (same length as texts1).
            batch_size: Mini-batch size to control memory usage.
            max_length: Truncation length for the tokenizer.
            return_probs: If True, also return entailment probabilities.

        Returns:
            preds: List[int] with MNLI class indices for each pair
                   (0=contradiction, 1=neutral, 2=entailment).
        """
        preds = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            n = len(texts1)
            for i in range(0, n, batch_size):
                batch1 = texts1[i : i + batch_size]
                batch2 = texts2[i : i + batch_size]

                # Tokenize a batch of (premise, hypothesis) pairs.
                enc = self.tokenizer(
                    batch1,
                    batch2,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                # Forward pass: logits shape [B, 3]
                logits = self.model(**enc).logits

                # Convert logits to probabilities over classes.
                probs = F.softmax(logits, dim=1)   # 0:contra, 1:neutral, 2:entail

                # Predicted class indices.
                pred = torch.argmax(probs, dim=1)  # shape [B]

                preds.extend(pred.cpu().tolist())

        return preds