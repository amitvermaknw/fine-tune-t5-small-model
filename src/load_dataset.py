import torch
from datasets import load_dataset
import evaluate

class LoadDataset:
    def load_data(self):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        # print("Dataset loaded", dataset)
        return dataset