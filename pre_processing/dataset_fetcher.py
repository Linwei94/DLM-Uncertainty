import pandas as pd
import random

class DatasetFetcher:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset: pd.DataFrame = self.get_dataset()

    def get_dataset(self):

        # Truthful QA
        # ------------------------------------------------------------------------------------
        if self.dataset_name == "truthful_qa":
            dataset = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")
            dataset["answer_key"] = dataset["Best Answer"]
            dataset["question"] = dataset["Question"]
        elif self.dataset_name == "truthful_qa_mini":
            dataset = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv").sample(n=100, random_state=42).reset_index(drop=True)
            dataset["answer_key"] = dataset["Best Answer"]
            dataset["question"] = dataset["Question"]


        # MMLU
        # ------------------------------------------------------------------------------------
        elif self.dataset_name == "mmlu":
            splits = {'test': 'all/test-00000-of-00001.parquet', 
                      'validation': 'all/validation-00000-of-00001.parquet', 
                      'dev': 'all/dev-00000-of-00001.parquet', 
                      'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
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
            splits = {'test': 'all/test-00000-of-00001.parquet', 
                      'validation': 'all/validation-00000-of-00001.parquet', 
                      'dev': 'all/dev-00000-of-00001.parquet', 
                      'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
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

        # Hellaswag
        # ------------------------------------------------------------------------------------
        elif self.dataset_name == "hellaswag":
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        elif self.dataset_name == "mmlu_mini":
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")

        # GPQA
        # ------------------------------------------------------------------------------------
        elif self.dataset_name == "gpqa":
            dataset = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")
            # Build formatted question and answer key
            questions, answer_keys = [], []
            for _, row in dataset.iterrows():
                # Step 1: Collect the answer choices
                choices = [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]

                # Step 2: Apply permutation (shuffle order based on the given permutation column)
                perm = random.sample(range(4), 4)
                choices = [choices[i] for i in perm]

                # Step 3: Determine correct answer letter (A/B/C/D)
                correct_index = choices.index(row["Correct Answer"])
                correct_answer = "ABCD"[correct_index]

                # Step 4: Build formatted question text
                question_text = (
                    f"{row['Question']}\n\n"
                    f"A. {choices[0].strip()}\n"
                    f"B. {choices[1].strip()}\n"
                    f"C. {choices[2].strip()}\n"
                    f"D. {choices[3].strip()}\n"
                    f"Answer: [Return Your Answer Letter ONLY]\n"
                )

                # Append to lists
                questions.append(question_text)
                answer_keys.append(correct_answer)

            # Add new columns to dataset
            dataset["question"] = questions
            dataset["answer_key"] = answer_keys

        elif self.dataset_name == "gpqa_mini":
            dataset = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv").sample(n=100, random_state=42).reset_index(drop=True)
            # Build formatted question and answer key
            questions, answer_keys = [], []
            for _, row in dataset.iterrows():
                # Step 1: Collect the answer choices
                choices = [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]

                # Step 2: Apply permutation (shuffle order based on the given permutation column)
                perm = random.sample(range(4), 4)
                choices = [choices[i] for i in perm]

                # Step 3: Determine correct answer letter (A/B/C/D)
                correct_index = choices.index(row["Correct Answer"])
                correct_answer = "ABCD"[correct_index]

                # Step 4: Build formatted question text
                question_text = (
                    f"{row['Question']}\n\n"
                    f"A. {choices[0].strip()}\n"
                    f"B. {choices[1].strip()}\n"
                    f"C. {choices[2].strip()}\n"
                    f"D. {choices[3].strip()}\n"
                    f"Answer: [Return Your Answer Letter ONLY]\n"
                )

                # Append to lists
                questions.append(question_text)
                answer_keys.append(correct_answer)

            # Add new columns to dataset
            dataset["question"] = questions
            dataset["answer_key"] = answer_keys

        # GSM8K
        # ------------------------------------------------------------------------------------
        elif self.dataset_name == "gsm8k":
            splits = {'train': 'socratic/train-00000-of-00001.parquet', 'test': 'socratic/test-00000-of-00001.parquet'}
            dataset = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])
            print(dataset)
            dataset["answer_key"] = dataset["answer"].str.split("####").str.get(1).str.strip()

        elif self.dataset_name == "gsm8k_mini":
            splits = {'train': 'socratic/train-00000-of-00001.parquet', 'test': 'socratic/test-00000-of-00001.parquet'}
            dataset = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"]).sample(n=100, random_state=42).reset_index(drop=True)
            dataset["answer_key"] = dataset["answer"].str.split("####").str.get(1).str.strip()
        
        # MMLU-Pro
        # ------------------------------------------------------------------------------------
        elif self.dataset_name == "mmlu_pro":
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        elif self.dataset_name == "mmlu_pro_mini":
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        
        # ARC-C
        # ------------------------------------------------------------------------------------
        elif self.dataset_name == "arcc":
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        elif self.dataset_name == "arcc_mini":
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        

        # ------------------------------------------------------------------------------------
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        
        return dataset
    

if __name__ == "__main__":
    df = DatasetFetcher(dataset_name="gsm8k").dataset
    idx = 2
    print(df["question"][idx])
    print(df["answer_key"][idx])