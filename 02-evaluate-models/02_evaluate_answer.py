import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.prompt.prompt_manager import PromptTemplateManager
from libs.Evaluation.qa_evaluation_with_llm import QAEvaluationLLM
from libs.hugging_face.DatasetManager import HuggingFaceDatasetManager

# Load environment variables
load_dotenv()
model_path = os.getenv("Bio_GPT_Large")
evaluator = QAEvaluationLLM(model_path)

# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)
dataset = load_dataset("myothiha/ontobench_path_vqa_result", split="train", cache_dir="/mnt/synology/myothiha/HF_CACHE")

results = []
for i, row in enumerate(dataset):
    question = row["question"]
    ref_answer = row["answer"]
    model_answer = row["model_answer"]

    result = evaluator.evaluate(question, ref_answer, model_answer)

    print("Extracted JSON:", result)
    row["score"] = result["score"]
    row["rationale"] = result

    results.append(row)

hf_DatasetManager = HuggingFaceDatasetManager(data=results, repo_name="myothiha/ontobench_path_vqa_result", hf_token=hf_token)
hf_DatasetManager.push()