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
from libs.dataset_loader.pathvqa_dataset_loader import PathVQADatasetLoader

from libs.question_classifier import LLMBasedQuestionClassifier

# Load environment variables
load_dotenv()

# Instantiate the question classifier
# Path to the saved Roberta model and tokenizer
classifier_model_dir = os.path.join(os.path.dirname(__file__), "classifier/roberta")
question_classifier = LLMBasedQuestionClassifier(classifier_model_dir)

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")
extractor = MedicalConceptExtractor()

# Input file
csv_input = "results/03_generate_vlm_reasoning_questions.csv"

# Load reasoning questions
df = pd.read_csv(csv_input).set_index('class')
reasoning_questions = df.to_dict(orient="index")


new_data = []
for row in vqa_loader.sample(n=1000, seed=42):
    img1 = imagehash.phash(row['image_path'])
    question = row["question"]
    answer = row["answer"]
    combined_text = f"{question} {answer}"
    concepts = extractor.extract(combined_text)
    concepts = [concept.text for concept in concepts]
    print(concepts)
    multiconcept_reasoning_questions = dict()
    for concept in concepts:
        try:
            vlm_reasoning_questions = json.loads(reasoning_questions[concept]['vlm_reasoning_questions'])

            for question in vlm_reasoning_questions.get('no_questions', []):
                for c in concepts:
                    if c!=concept and question_classifier.classify(c, question) == 1:
                        print(f"Removing question '{question}' for concept '{concept}' as they are relevant.")
                        vlm_reasoning_questions['no_questions'].remove(question)
        except:
            continue
        multiconcept_reasoning_questions[concept] = vlm_reasoning_questions

    row['concepts'] = json.dumps(concepts)
    row['multiconcept_reasoning_questions'] = json.dumps(multiconcept_reasoning_questions)
    new_data.append(row)

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