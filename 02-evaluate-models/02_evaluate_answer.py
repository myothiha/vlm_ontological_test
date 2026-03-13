import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.Evaluation.qa_evaluation_with_llm import QAEvaluationLLM
from libs.hugging_face.DatasetManager import HuggingFaceDatasetManager
from models_config import models

# Load environment variables
load_dotenv()
# model_path = os.getenv("Bio_GPT_Large")
llm = OllamaWrapper(model="gpt-oss:20b")
evaluator = QAEvaluationLLM(llm)

# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)
cache_dir = os.getenv("HF_CACHE_DIR")

for model_name in models.keys():

    print(f"Evaluate Answers for {model_name}")

    dataset_path = f"myothiha/conceptbench_path_vqa_result_2_{model_name}"
    dataset = load_dataset(dataset_path, split="train", cache_dir=cache_dir)

    results = []
    for i, row in enumerate(dataset):
        question = row["question"]
        ref_answer = row["answer"]
        model_answer = row["model_answer"]

        result = evaluator.evaluate(question, ref_answer, model_answer)

        print("Extracted JSON:", result)
        row["model_answer_accuracy"] = result["score"]
        row["model_answer_evaluation_details"] = result

        results.append(row)

    dataset_path += "_evaluated"

    hf_DatasetManager = HuggingFaceDatasetManager(data=results, repo_name=dataset_path, hf_token=hf_token)
    hf_DatasetManager.push()