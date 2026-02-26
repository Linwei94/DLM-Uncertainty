
import gc
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from torch.nn import functional as F
from default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from default_utils.registry import register_confidence

class EntailmentDeberta():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli", load_in_8bit=True, device_map="auto")

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
        batch_size: int = 512,
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
        device = next(self.model.parameters()).device
        
        with torch.inference_mode():
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
                )
                
                # Move to device only once per batch
                enc = {k: v.to(device) for k, v in enc.items()}

                # Forward pass
                logits = self.model(**enc).logits

                # Predicted class indices.
                pred = torch.argmax(logits, dim=1)  # shape [B]

                preds.extend(pred.cpu().tolist())

        return preds


def semantic_uncertainty_selection(output_lst: list[ModelOutputs], **kwargs) -> tuple[list[str], list[float]]:
    response_lists: list[tuple[str]] = list(zip(*[outputs.output_texts for outputs in output_lst]))
    entailment_model = EntailmentDeberta()
    strict_entailment: bool = False
    selected_responses = []
    confidences = []
    for response_set in tqdm(response_lists, desc="Processing Entailments (Semantic Groups)"):
        n = len(response_set)

        if n == 1:
            semantic_ids = [0]
        else:
            # -------------------------------
            # Build bidirectional entailment pairs
            # -------------------------------
            left, right = [], []
            pair_keys = []

            for i in range(n):
                for j in range(i + 1, n):
                    # i -> j
                    left.append(response_set[i])
                    right.append(response_set[j])
                    pair_keys.append((i, j, "ij"))

                    # j -> i
                    left.append(response_set[j])
                    right.append(response_set[i])
                    pair_keys.append((i, j, "ji"))

            # Run batch entailment
            batch_results = entailment_model.check_implication_batch(left, right)

            # Map results to dictionary
            entail_map = {}
            for (i, j, direction), res in zip(pair_keys, batch_results):
                entail_map.setdefault((i, j), {})[direction] = res

            # -------------------------------
            # Semantic equivalence
            # -------------------------------
            def are_equivalent(i, j):
                ij = entail_map[(i, j)]["ij"]
                ji = entail_map[(i, j)]["ji"]
                if strict_entailment:
                    return ij == 2 and ji == 2
                else:
                    # Paper non-strict: no contradiction, not both neutral
                    return (ij != 0) and (ji != 0) and not (ij == 1 and ji == 1)

            # -------------------------------
            # Assign semantic IDs
            # -------------------------------
            semantic_ids = [-1] * n
            next_id = 0
            for i in range(n):
                if semantic_ids[i] == -1:
                    semantic_ids[i] = next_id
                    for j in range(i + 1, n):
                        if are_equivalent(i, j):
                            semantic_ids[j] = next_id
                    next_id += 1

        # -------------------------------
        # Step 2: Find most frequent semantic ID
        # -------------------------------
        most_freq_id = max(set(semantic_ids), key=semantic_ids.count)
        confidence = semantic_ids.count(most_freq_id) / n
        confidences.append(confidence)

        # -------------------------------
        # Step 3: Pick first response with that semantic ID
        # -------------------------------
        selected_response = response_set[semantic_ids.index(most_freq_id)]
        selected_responses.append(selected_response)

        
    # entailment_model.model.to("cpu")
    del entailment_model.model
    del entailment_model.tokenizer
    del entailment_model
    gc.collect()
    torch.cuda.empty_cache()
    return selected_responses, confidences


def semantic_cluster_selection(output_lst: list[ModelOutputs], **kwargs) -> tuple[list[list[str]], list[list[list[float]]]]:
    response_lists: list[tuple[str]] = list(zip(*[outputs.output_texts for outputs in output_lst]))
    logprobs_lists: list[tuple[list[float]]] = list(zip(*[outputs.output_logprobs for outputs in output_lst]))
    entailment_model = EntailmentDeberta()
    strict_entailment: bool = False
    clusters: list[list[str]] = []
    clusters_logprobs: list[list[list[float]]] = []
    for idx, response_set in enumerate(tqdm(response_lists, desc="Processing Entailments (Semantic Groups)")):
        logprobs_set = logprobs_lists[idx]
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

        # Step 2: Find most frequent semantic ID and collect all its members
        most_freq_id = max(set(semantic_ids), key=semantic_ids.count)
        cluster_indices = [i for i, sid in enumerate(semantic_ids) if sid == most_freq_id]
        cluster_members = [response_set[i] for i in cluster_indices]
        cluster_members_logprobs = [logprobs_set[i] for i in cluster_indices]
        clusters.append(cluster_members)
        clusters_logprobs.append(cluster_members_logprobs)
        
    entailment_model.model.to("cpu")
    del entailment_model.model
    del entailment_model.tokenizer
    del entailment_model
    gc.collect()
    torch.cuda.empty_cache()
    return clusters, clusters_logprobs


@register_confidence(name="semantic_uncertainty")
def semantic_uncertainty(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs):
    selected_responses, confidences = semantic_uncertainty_selection(output_lst, **kwargs)
    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[confidences]
    )