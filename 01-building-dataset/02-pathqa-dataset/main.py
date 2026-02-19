import os
import sys
from dotenv import load_dotenv

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.concept_extractor.medical_concept_classifier import MedicalConceptClassifier
from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.OntologyBenchmarkBuilder.benchmark_builder import BenchmarkBuilder
from libs.dataset_loader.dataset import Dataset
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.utils import extract_list_from_gpt, extract_list_from_ollama_models, extract_list_from_hf_models, setup_logger

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")

# Setup logger
logger = setup_logger("Main", "results/main.log")
logger.info("Starting dataset building...")

# Load environment variables
load_dotenv()
# model_path = os.getenv("Llama3_OpenBioLLM_70B")
model_path = "gpt-oss:20b"

# Load model

# HF Model
# llm = LLMWrapper(model_path=model_path, quantization_bits=4)

# Use Ollama LLM as api. There is a docker server running Ollama with the model.
# llm = OllamaWrapper(model=model_path)

# api model
llm = GPTLLMWrapper("gpt-4.1")


# extractor = MedicalConceptExtractor("en_core_sci_scibert") # Load the SciBERT model for medical concept extraction

medical_concept_classifier = MedicalConceptClassifier(OllamaWrapper(model="gpt-oss:20b"))
extractor = LLMBasedMedicalConceptExtractor(model=OllamaWrapper(model="gpt-oss:20b"),
                                            backup_extractor=MedicalConceptExtractor("en_core_sci_scibert"),
                                            concept_classifier=medical_concept_classifier)  # Load the LLM for medical concept extraction


unique_concepts = set()

# dataset = vqa_loader.get_all()
dataset = vqa_loader.sample(n=10)

benchmarkBuilder = BenchmarkBuilder(
    dataset=dataset,
    llm=llm,
    result_extract_func=extract_list_from_gpt,
    concept_extractor=extractor,
    required_concept_extraction=True,
    medical_concept_classifier=medical_concept_classifier,
    output_dir="results",
    knowledge_question_prompt_templates="prompt_templates/kq_prompt_templates",
    generate_knowledge_prompt_template="prompt_templates/ck_prompt_templates",
    vlm_reasoning_prompt_template="prompt_templates/rq_prompt_templates",
)

vlm_reasoning_questions_csv = benchmarkBuilder.build()

