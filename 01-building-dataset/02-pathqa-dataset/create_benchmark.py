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

# Load environment variables
load_dotenv()
model_path = os.getenv("Llama3_8B_Instruct")

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")
extractor = MedicalConceptExtractor()

# Load the VLM reasoning questions dataset
vlm_reasoning_questions_csv = "results/03_generate_vlm_reasoning_questions.csv"
reasoningDataset = VLMReasoningQuestionsLoader(vlm_reasoning_questions_csv, llm_model_path=model_path)

new_data = []
for row in vqa_loader.sample(n=5):
    img1 = imagehash.phash(row['image'])
    questions_and_answers = row['questions_and_answers']
    text = row['text']
    all_concepts = extractor.extract(text)
    
    all_concepts = [concept.text for concept in all_concepts]
    all_concepts = list(set(all_concepts))  # Remove duplicates
    print("all_concepts:", all_concepts)

    for question_answer in questions_and_answers:
        
        question = question_answer['question']
        answer = question_answer['answer']

        # Extract concepts from the question and answer
        combined_text = f"{question} {answer}"
        current_concepts = extractor.extract(combined_text)
        print("current_concepts:", current_concepts)

        # Combine concepts from both question and answer
        current_concepts = [concept.text for concept in current_concepts]
        current_concepts = list(set(current_concepts))  # Remove duplicates
    
        multiconcept_reasoning_questions = dict()
        for concept in current_concepts:

            try:
                exclude_concepts = [c for c in all_concepts if c != concept]
                vlm_reasoning_questions = {
                    "yes_questions": reasoningDataset.get_positive_questions(concept), 
                    "no_questions": reasoningDataset.get_negative_questions(concept, exclude_concepts=exclude_concepts, num_concepts=1, num_questions=3),
                }

                multiconcept_reasoning_questions[concept] = vlm_reasoning_questions
            except Exception as e:
                print(f"Error {e}")
                continue

        new_row = dict()
        new_row["image"] = row['image']
        new_row['question'] = question
        new_row['answer'] = answer
        new_row['related_concepts'] = current_concepts
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
    new_dataset.push_to_hub("myothiha/ontobench_path_vqa")
    print("✅ Dataset pushed successfully to Hugging Face.")

except Exception as e:
    print("❌ Failed to push dataset:", e)
    print("⚠️ Keeping local files for debugging.")