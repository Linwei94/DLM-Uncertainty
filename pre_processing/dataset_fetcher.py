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
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        
        return dataset