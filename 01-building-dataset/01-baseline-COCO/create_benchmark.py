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
from libs.dataset_loader.coco_dataset_loader import COCOLoader
from libs.OntologyBenchmarkBuilder.vlm_reasoning_test import VLMReasoningQuestionsLoader

# Load environment variables
load_dotenv()

# Load COCO dataset
dataset_loader = COCOLoader()

# Load the VLM reasoning questions dataset
vlm_reasoning_questions_csv = "results/03_generate_vlm_reasoning_questions.csv"
reasoningDataset = VLMReasoningQuestionsLoader(vlm_reasoning_questions_csv)

new_data = []
for row in dataset_loader.sample_original_dataset(n=1000, seed=42):
    concepts = row["text"]

    multiconcept_reasoning_questions = dict()
    for concept in concepts:
        exclude_concepts = [c for c in concepts if c != concept]
        vlm_reasoning_questions = {
            "yes_questions": reasoningDataset.get_positive_questions(concept), 
            "no_questions": reasoningDataset.get_negative_questions(concept, exclude_concepts=exclude_concepts),
        }

        multiconcept_reasoning_questions[concept] = vlm_reasoning_questions

    row['concepts'] = json.dumps(concepts)
    row['multiconcept_reasoning_questions'] = json.dumps(multiconcept_reasoning_questions)
    del row['text']  # Remove the original text column to avoid redundancy
    new_data.append(row)

# Create new dataset with the new column
new_dataset = Dataset.from_list(new_data)

# STEP 1: Authenticate
login(token=os.getenv("HF_ACCESS_TOKEN"))  # Or just `login()` to log in interactively

try:
    # STEP 5: Push to Hub
    new_dataset.push_to_hub("myothiha/ontobench_coco")
    print("✅ Dataset pushed successfully to Hugging Face.")

except Exception as e:
    print("❌ Failed to push dataset:", e)
    print("⚠️ Keeping local files for debugging.")