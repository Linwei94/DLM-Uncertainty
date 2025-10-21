import pandas as pd

class PromptFormatter:
    def __init__(self, dataset:pd, prompt_type:str = "qa"):
        self.dataset: pd.DataFrame = dataset
        self.prompt_type: str = prompt_type
        self.prompts: list[str] = self.format_prompts(prompt_type)

    def format_prompts(self, prompt_type):
        if prompt_type == "qa":
            prompt = """
            Question: {question}
            Answer:
            """.strip()
            qa_prompts = self.dataset["question"].apply(lambda x: prompt.format(question=x)).tolist()
            return qa_prompts
        if prompt_type == "vanilla":
            prompt = """
            Answer the following question using a succinct (at most one sentence) and full answer.

            Question: {question}
            Answer:
            """.strip()
            vanilla_prompts = self.dataset["question"].apply(lambda x: prompt.format(question=x)).tolist()
            return vanilla_prompts
        elif prompt_type == "vanilla_uncertainty":
            prompt = """
            Answer the following question using a succinct (at most one sentence) and full answer. 
            If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

            Question: {question}
            Answer:
            """.strip()
            vanilla_uncertainty_prompts = self.dataset["question"].apply(lambda x: prompt.format(question=x)).tolist()
            return vanilla_uncertainty_prompts
        elif prompt_type == "vnc":
            prompt = """
            Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
            {question}
            Please provide a confidence score between 0 and 100 at the end of your answer in the following JSON format:
            {{
                "answer": Your answer here,
                "confidence_score": number
            }}
            """.strip()
            vnc_prompts = self.dataset["question"].apply(lambda x: prompt.format(question=x)).tolist()
            return vnc_prompts
        elif prompt_type == "mmlu_0":
            mmlu_0_prompts = self.dataset["question"]
            return mmlu_0_prompts
        elif prompt_type == "mmlu_5":
            mmlu_5_prompts = self.dataset["few_shot"] + self.dataset["question"]
            return mmlu_5_prompts
        elif prompt_type == "mmlu_vnc":
            mmlu_vnc = """
            Read the question, provide your answer and your confidence in this
            answer. Note: The confidence indicates how likely you think your
            answer is true.
            Use the following format to answer:

            “Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%”

            Only the answer and confidence, don't give me the explanation.
            Question:[{question}]
            Now, please answer this question and provide your confidence level.

            """
            return [mmlu_vnc.format(question=q) for q in self.dataset["question"]]
        else:
            raise NotImplementedError(f"Prompt type {self.prompt_type} not implemented.")