# main.py
import torch
import os
import sys

from PIL import Image
import imagehash

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gc
import json
import pandas as pd

from dotenv import load_dotenv
from collections import defaultdict

from PIL import Image

from datasets import load_dataset, Dataset, Features, Value, Image
from huggingface_hub import login
import shutil

from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.OntologyBenchmarkBuilder.vlm_reasoning_test import VLMReasoningQuestionsLoader
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.concept_extractor.medical_concept_classifier import MedicalConceptClassifier
from libs.OntologyBenchmarkBuilder.pipeline import OntoBenchPipeline
from libs.utils import setup_logger

# Load environment variables
load_dotenv()
model_path_for_similarity = os.getenv("Llama3_8B_Instruct")

# setup logger
logger = setup_logger("CreateBenchmark", os.path.join("results", "create_benchmark.log"))

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")

# extractor = MedicalConceptExtractor()
medical_concept_classifier = MedicalConceptClassifier(OllamaWrapper(model="gpt-oss:20b"))
concept_extractor = LLMBasedMedicalConceptExtractor(model=OllamaWrapper(model="gpt-oss:20b"),
                                            backup_extractor=MedicalConceptExtractor("en_core_sci_scibert"),
                                            concept_classifier=medical_concept_classifier)  # Load the LLM for medical concept extraction

# Setup pipeline to generate questions for concepts that are not exist.
llm = GPTLLMWrapper("gpt-4.1")
# llm = OllamaWrapper(model="gpt-oss:20b")

concept_bench_pipeline = OntoBenchPipeline(
    llm = llm, # Use for negative questions generation. LLM encoder to check most dissimilar concepts. None means randomly sample.
    knowledge_question_prompt_templates = "prompt_templates/kq_prompt_templates",
    generate_knowledge_prompt_template = "prompt_templates/ck_prompt_templates",
    vlm_reasoning_prompt_template = "prompt_templates/rq_prompt_templates"
)

# Load the VLM reasoning questions dataset
vlm_reasoning_questions_csv = "results/03_generate_vlm_reasoning_questions.csv"
reasoningDataset = VLMReasoningQuestionsLoader(
    vlm_reasoning_questions_csv,
    concept_bench=concept_bench_pipeline,
    # llm_model_path=model_path_for_similarity # For selecting dissimilar concepts for negative questions. Comment out for random sampling.
)

new_data = []
for row in vqa_loader.sample(n=1):
    img1 = imagehash.phash(row['image'])
    questions_and_answers = row['questions_and_answers']
    text = row['text']
    
    # all_concepts = [concept.text for concept in all_concepts]
    all_concepts = set()  # Remove duplicates

    for question_answer in questions_and_answers:
        
        question = question_answer['question']
        answer = question_answer['answer']

        # Extract concepts from the question and answer
        combined_text = f"{question} {answer}"
        current_concepts = concept_extractor.extract(combined_text)

        # Combine concepts from both question and answer
        # current_concepts = [concept.text for concept in current_concepts]
        current_concepts = list(set(current_concepts))  # Remove duplicates

        question_answer['extracted_concepts'] = current_concepts
        all_concepts.update(current_concepts)
    
    multiconcept_reasoning_questions = dict()
    
    all_concepts = list(set(all_concepts)) # Remove duplicates

    print("All concepts", all_concepts)
    for concept in all_concepts:

        try:
            exclude_concepts = [c for c in all_concepts if c != concept]
            vlm_reasoning_questions = {
                "yes_questions": reasoningDataset.get_positive_questions(concept), 
                "no_questions": reasoningDataset.get_negative_questions(concept, exclude_concepts=exclude_concepts, num_concepts=10, num_questions=15),
            }

            logger.info(f"Concept: {concept}")
            logger.info(f"Yes Questions: {len(vlm_reasoning_questions['yes_questions'])}")
            logger.info(f"No Questions: {len(vlm_reasoning_questions['no_questions'])}")

            multiconcept_reasoning_questions[concept] = vlm_reasoning_questions
        except Exception as e:
            print(f"Questions Generation Error: {e}")
            continue

    new_row = dict()
    new_row["image"] = row['image']
    new_row['questions_and_answers'] = questions_and_answers
    new_row['all_concepts'] = json.dumps(all_concepts)
    new_row['multiconcept_reasoning_questions'] = json.dumps(multiconcept_reasoning_questions)
    print("new_row:", new_row)
    new_data.append(new_row)

print("New Data", new_data)
# Create new dataset with the new column
new_dataset = Dataset.from_list(new_data)

# STEP 1: Authenticate
login(token=os.getenv("HF_ACCESS_TOKEN"))  # Or just `login()` to log in interactively

try:
    # STEP 5: Push to Hub
    new_dataset.push_to_hub("myothiha/conceptbench_path_vqa")
    print("✅ Dataset pushed successfully to Hugging Face.")

except Exception as e:
    print("❌ Failed to push dataset:", e)
    print("⚠️ Keeping local files for debugging.")