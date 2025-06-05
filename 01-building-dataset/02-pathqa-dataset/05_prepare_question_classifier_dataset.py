# main.py

import torch
import os
import sys

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

# Load environment variables
load_dotenv()

# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")

extractor = MedicalConceptExtractor()

# Input file
csv_input = "results/03_generate_vlm_reasoning_questions.csv"

# Output file setup

# Load reasoning questions
df = pd.read_csv(csv_input).set_index('class')
reasoning_questions = df.to_dict(orient="index")

# output file for concept question labels
csv_filename = "results/05_concept_question_labels.csv"
pd.DataFrame(columns=["class", "questons", "label"]).to_csv(csv_filename, index=False)

rows = []
for concept_name, row in reasoning_questions.items():
    try:
        questions_json = json.loads(row['vlm_reasoning_questions'])
        yes_questions = questions_json.get('yes_questions', [])
        no_questions = questions_json.get('no_questions', [])
        for q in yes_questions:
            rows.append({"class": concept_name, "questons": q, "label": 1})
        for q in no_questions:
            rows.append({"class": concept_name, "questons": q, "label": 0})
    except Exception as e:
        print(f"Failed to parse questions for {concept_name}: {e}")

if rows:
    pd.DataFrame(rows).to_csv(csv_filename, mode='a', index=False, header=False)
    print(f"✅ Saved concept-question-label CSV to {csv_filename}")
else:
    print("No data to save.")