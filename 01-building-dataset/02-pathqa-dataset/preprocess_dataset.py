import os
import sys
from dotenv import load_dotenv
from collections import defaultdict

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.dataset_loader.pathvqa_dataset_loader import PathVQADatasetLoader
from libs.OntologyBenchmarkBuilder.benchmark_builder import BenchmarkBuilder
from datasets import load_dataset, Dataset, Features, Value, Image
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.utils import extract_list_from_gpt

from huggingface_hub import login

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")

# extractor = MedicalConceptExtractor("en_core_sci_scibert") # Load the SciBERT model for medical concept extraction

# Load environment variables
load_dotenv()
model_path = os.getenv("Llama3_OpenBioLLM_70B")

# Load model
# llm = LLMWrapper(model_path=model_path, quantization_bits=4)
llm = GPTLLMWrapper("gpt-4.1")

unique_concepts = set()

dataset = vqa_loader.get_all()

results = {}
for item in dataset:
    hash = item["image_hash"]
    qestions_and_answers = {
        "question": item["question"],
        "answer": item["answer"],
    }
    
    if hash not in results:
        results[hash] = {
            "image": item["image"],
            "questions_and_answers": []
        }
            
    results[hash]["questions_and_answers"].append(qestions_and_answers)


new_data = []
for result in results.values():
    new_data.append({
        "image": result["image"],
        "questions_and_answers": result["questions_and_answers"]
    })

# Create new dataset with the new column
new_dataset = Dataset.from_list(new_data)

# STEP 1: Authenticate
login(token=os.getenv("HF_ACCESS_TOKEN"))  # Or just `login()` to log in interactively

try:
    # STEP 5: Push to Hub
    new_dataset.push_to_hub("myothiha/processed_pathvqa")
    print("✅ Dataset pushed successfully to Hugging Face.")

except Exception as e:
    print("❌ Failed to push dataset:", e)
    print("⚠️ Keeping local files for debugging.")