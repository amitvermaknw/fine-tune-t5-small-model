import torch
from datasets import load_dataset
import evaluate

class LoadDataset:
    def load_data(self):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        # For testing on vertex ai using the 1000 dataset for training
        # And 200 dataset for val
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(1000))

        return dataset