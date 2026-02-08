import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.Evaluation.qa_evaluation_with_llm import QAEvaluationLLM
from libs.hugging_face.DatasetManager import HuggingFaceDatasetManager

# Load environment variables
load_dotenv()
# model_path = os.getenv("Bio_GPT_Large")
llm = OllamaWrapper(model="gpt-oss:20b")
evaluator = QAEvaluationLLM(llm)

# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)

models = [
    # "llava_34b",
    # "gemma3_27b",
    # "llava_med_v1.5_mistral_7b",
    # "mistral_small3.2_24b",
    # "qwen2.5vl_72b",
    "llama3.2_vision_11b_instruct",
]

for model_name in models:

    print(f"Evaluate Knowledge Answers for {model_name}")

    dataset_path = f"myothiha/conceptbench_path_vqa_result_2_{model_name}_evaluated_ICL"
    dataset = load_dataset(dataset_path, split="train", cache_dir="/mnt/synology/myothiha/HF_CACHE")

    results = []
    for i, row in enumerate(dataset):
        question = row["question"]
        ref_answer = row["answer"]
        knowledge_answer = row["knowledge_answer"]

        result = evaluator.evaluate(question, ref_answer, knowledge_answer)

        print("Extracted JSON:", result)
        row["knowledge_answer_accuracy"] = result["score"]
        row["knowledge_answer_evaluation_details"] = result

        results.append(row)

    dataset_path += "_evaluated"

    hf_DatasetManager = HuggingFaceDatasetManager(data=results, repo_name=dataset_path, hf_token=hf_token)
    hf_DatasetManager.push()