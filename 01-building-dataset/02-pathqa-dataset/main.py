import os
import sys
from dotenv import load_dotenv

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.OntologyBenchmarkBuilder.benchmark_builder import BenchmarkBuilder
from libs.dataset_loader.dataset import Dataset
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.utils import extract_list_from_gpt, extract_list

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")

# Load environment variables
load_dotenv()
model_path = os.getenv("Llama3_OpenBioLLM_70B")

# Load model
llm = LLMWrapper(model_path=model_path, quantization_bits=4)
# llm = GPTLLMWrapper("gpt-4.1")

# extractor = MedicalConceptExtractor("en_core_sci_scibert") # Load the SciBERT model for medical concept extraction
extractor = LLMBasedMedicalConceptExtractor(model=llm)  # Load the LLM for medical concept extraction

unique_concepts = set()

dataset = vqa_loader.sample(n=20)

benchmarkBuilder = BenchmarkBuilder(
    dataset=dataset,
    llm=llm,
    result_extract_func=extract_list,
    concept_extractor=extractor,
    required_concept_extraction=True,
    output_dir="results",
    knowledge_question_prompt_template="01_few_shot_without_instruction",
    generate_knowledge_prompt_template="02_generate_knowledge_prompt1",
    vlm_reasoning_prompt_template="03_vlm_reasoning_questions_one_shot",
)

vlm_reasoning_questions_csv = benchmarkBuilder.build()