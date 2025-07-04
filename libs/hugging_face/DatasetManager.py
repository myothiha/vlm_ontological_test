from datasets import Dataset
from huggingface_hub import login
import os

class HuggingFaceDatasetManager:
    def __init__(self, data: list, repo_name: str, hf_token=None):
        """
        Optionally provide a Hugging Face token for authentication.
        """
        self.data = data
        self.repo_name = repo_name
        if hf_token is None:
            hf_token = os.getenv("HF_ACCESS_TOKEN")
        login(token=hf_token)

    def push(self, private=False, dataset_split="train"):
        """
        Upload a list of dictionaries as a dataset to Hugging Face Hub.
        :param data: List of dictionaries (each dict is a row/sample)
        :param repo_name: Name of the dataset repo on Hugging Face (e.g., 'username/dataset_name')
        :param private: Whether to make the dataset private (default: False)
        :return: The URL of the uploaded dataset
        """
        dataset = Dataset.from_list(self.data)

        try:
            # STEP 5: Push to Hub
            url = dataset.push_to_hub(self.repo_name, private=private)
            print(f"✅ Dataset pushed successfully to Hugging Face. {url}")
        except Exception as e:
            print("❌ Failed to push dataset:", e)
            print("⚠️ Keeping local files for debugging.")
        return url