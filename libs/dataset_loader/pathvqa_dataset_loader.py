import os
import random
from datasets import load_dataset

class PathVQADatasetLoader:
    """
    Loader for the PathVQA dataset from Hugging Face.
    https://huggingface.co/datasets/flaviagiammarino/path-vqa
    """
    def __init__(self, split="train", data_dir=None):
        """
        Args:
            split (str): Which split to load ('train', 'validation', 'test').
            data_dir (str, optional): Directory to cache the dataset.
        """
        self.split = split
        self.data_dir = data_dir
        self.dataset = load_dataset(
            "flaviagiammarino/path-vqa",
            split=split,
            cache_dir=data_dir
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "image_path": item["image"]
        }

    def get_all(self):
        """Return all items as a list of dicts."""
        return [self[i] for i in range(len(self))]

    def sample(self, n=1, seed=None):
        """Return n random rows from the dataset as a list of dicts."""
        if seed is not None:
            random.seed(seed)
        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self[i] for i in indices]
