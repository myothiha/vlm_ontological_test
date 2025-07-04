import os
import sys
from dotenv import load_dotenv

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.dataset_loader.coco_dataset_loader import COCOLoader
from libs.OntologyBenchmarkBuilder.benchmark_builder import BenchmarkBuilder
from libs.dataset_loader.dataset import Dataset
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.utils import extract_list_from_gpt

# Load PathVQA dataset
dataset_loader = COCOLoader()  # Assuming COCOLoader is similar to PathVQADatasetLoader

# Load environment variables
load_dotenv()
# model_path = os.getenv("Llama3_OpenBioLLM_70B")

# Load model
# llm = LLMWrapper(model_path=model_path, quantization_bits=4)
llm = GPTLLMWrapper("gpt-4.1")

unique_concepts = set()

dataset = dataset_loader.sample(n=2000, seed=42)

benchmarkBuilder = BenchmarkBuilder(
    dataset=dataset,
    llm=llm,
    result_extract_func=extract_list_from_gpt,
    required_concept_extraction=False,
    output_dir="results",
    knowledge_question_prompt_template="01_generate_knowledge_questions_one_shot",
    generate_knowledge_prompt_template="02_generate_knowledge_prompt1",
    vlm_reasoning_prompt_template="03_positive_vlm_reasoning_questions_few_shots",
)

vlm_reasoning_questions_csv = benchmarkBuilder.build()