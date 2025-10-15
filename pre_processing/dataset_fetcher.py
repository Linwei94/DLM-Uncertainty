import pandas as pd

class DatasetFetcher:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset: pd.DataFrame = self.get_dataset()

    def get_dataset(self):
        if self.dataset_name == "truthful_qa":
            dataset = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")
            dataset["answer_key"] = dataset["Best Answer"]
            dataset["question"] = dataset["Question"]
        elif self.dataset_name == "truthful_qa_mini":
            dataset = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv").sample(n=15, random_state=42).reset_index(drop=True)
            dataset["answer_key"] = dataset["Best Answer"]
            dataset["question"] = dataset["Question"]
        if self.dataset_name == "mmlu":
            splits = {'test': 'all/test-00000-of-00001.parquet', 'validation': 'all/validation-00000-of-00001.parquet', 'dev': 'all/dev-00000-of-00001.parquet', 'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
            dataset = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
            few_shots_df = pd.read_parquet("hf://datasets/cais/mmlu/" + "all/dev-00000-of-00001.parquet")
            few_shot_blocks = []
            eval_questions = []
            for i, row in dataset.iterrows():
                subject = row["subject"]
                examples = few_shots_df[few_shots_df["subject"] == subject]
                few_shot_block = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
                for _, ex in examples.iterrows():
                    few_shot_block += (
                        f"Question: {ex['question']}\n"
                        f"A. {ex['choices'][0]}\n"
                        f"B. {ex['choices'][1]}\n"
                        f"C. {ex['choices'][2]}\n"
                        f"D. {ex['choices'][3]}\n"
                        f"Answer: {chr(ord("A") + ex['answer'])}\n\n"
                    )
                eval_question = (
                    f"Question: {row['question']}\n"
                    f"A. {row['choices'][0]}\n"
                    f"B. {row['choices'][1]}\n"
                    f"C. {row['choices'][2]}\n"
                    f"D. {row['choices'][3]}\n"
                    f"Answer: [Return Your Answer Letter ONLY]\n"
                )
                few_shot_blocks.append(few_shot_block)
                eval_questions.append(eval_question)

            dataset["few_shot"] = few_shot_blocks
            dataset["question"] = eval_questions
            dataset["answer_key"] = (dataset["answer"]).apply(lambda x: chr(ord("A") + x))

        elif self.dataset_name == "mmlu_mini":
            splits = {'test': 'all/test-00000-of-00001.parquet', 'validation': 'all/validation-00000-of-00001.parquet', 'dev': 'all/dev-00000-of-00001.parquet', 'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
            dataset = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"]).sample(n=100, random_state=42).reset_index(drop=True)
            few_shots_df = pd.read_parquet("hf://datasets/cais/mmlu/" + "all/dev-00000-of-00001.parquet")
            few_shot_blocks = []
            eval_questions = []
            for i, row in dataset.iterrows():
                subject = row["subject"]
                examples = few_shots_df[few_shots_df["subject"] == subject]
                few_shot_block = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
                for _, ex in examples.iterrows():
                    few_shot_block += (
                        f"Question: {ex['question']}\n"
                        f"A. {ex['choices'][0]}\n"
                        f"B. {ex['choices'][1]}\n"
                        f"C. {ex['choices'][2]}\n"
                        f"D. {ex['choices'][3]}\n"
                        f"Answer: {chr(ord("A") + ex['answer'])}\n\n"
                    )
                eval_question = (
                    f"Question: {row['question']}\n"
                    f"A. {row['choices'][0]}\n"
                    f"B. {row['choices'][1]}\n"
                    f"C. {row['choices'][2]}\n"
                    f"D. {row['choices'][3]}\n"
                    f"Answer: [Return Your Answer Letter ONLY]\n"
                )
                few_shot_blocks.append(few_shot_block)
                eval_questions.append(eval_question)

            dataset["few_shot"] = few_shot_blocks
            dataset["question"] = eval_questions
            dataset["answer_key"] = (dataset["answer"]).apply(lambda x: chr(ord("A") + x))
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        
        return dataset