import os
from datasets import load_dataset
from libs.dataset_loader.dataset import Dataset
from PIL import Image
import imagehash

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
            "image_hash": self.get_image_hash_from_pil(item["image"]),
            "question": item["question"],
            "answer": item["answer"],
            "text": f"{item['question']} {item['answer']}",
            "image": item["image"]
        }

    def get_all(self):
        """Return all items as a list of dicts."""
        data = [self[i] for i in range(len(self))]
        dataset = Dataset(data)
        return dataset

    def sample(self, n=1, seed=None):
        """Return the first n rows from the dataset as a list of dicts."""
        data = [self[i] for i in range(min(n, len(self)))]
        dataset = Dataset(data)
        return dataset
    
    def sample_original_dataset(self, n=1, seed=None):
        """Return the first n rows from the original HuggingFace dataset as a list of dicts (raw format)."""
        data = [self.dataset[i] for i in range(min(n, len(self.dataset)))]
        return data

    def get_image_hash(self, idx, hash_func=None):
        """
        Compute a hash for the image at the given index. By default, uses phash from imagehash.
        :param idx: Index of the image in the dataset.
        :param hash_func: Optional custom hash function. If None, uses imagehash.phash.
        :return: Hash string for the image.
        """
        
        item = self.dataset[idx]
        img = item["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if hash_func is None:
            hash_func = imagehash.phash
        return str(hash_func(img))

    def get_image_hash_from_pil(self, pil_image, hash_func=None):
        """
        Compute a hash for a given PIL image. By default, uses phash from imagehash.
        :param pil_image: PIL.Image object.
        :param hash_func: Optional custom hash function. If None, uses imagehash.phash.
        :return: Hash string for the image.
        """
        
        if hash_func is None:
            hash_func = imagehash.phash
        return str(hash_func(pil_image))