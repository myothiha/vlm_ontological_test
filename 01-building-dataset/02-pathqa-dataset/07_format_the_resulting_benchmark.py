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

# Load environment variables
load_dotenv()

# Input file
csv_input = "results/03_generate_vlm_reasoning_questions.csv"

# Load reasoning questions
num_samples = 100
df = pd.read_csv(csv_input).set_index('class').sample(n=num_samples, random_state=42)

# print("Columns: ", df.columns.to_list())
dataset = df.to_dict(orient="index")

all_concepts = list(dataset.keys())
print(f"Selecting Random Concepts ({len(all_concepts)}): ", all_concepts)

conceptual_dimensions = [
    'contexual_properties', 
    'contexual_behavior', 
    'properties', 
    'functions',
    'relational', 
]

# target_concept = "bone"
target_concept = None
results = []
for concept, knowledge_source in dataset.items():
    if target_concept is not None:
        if concept != target_concept:
            continue
    
    text = ""

    text += "################################################################\n"
    text += f"CONCEPT: {concept}\n"
    text += "################################################################\n"

    # Generate knowledge questions.
    knowledge_questions = json.loads(knowledge_source['knowledge_questions'])
    conceptual_knowledge = json.loads(knowledge_source['generated_knowledge'])
    reasoning_questions = json.loads(knowledge_source['vlm_reasoning_questions'])
    
    # print("Knowledge Questions:\n", knowledge_questions)
    limits_for_piece_of_knowledge = 2
    for dimension in conceptual_dimensions:
        text += f"\nDimension: {dimension}\n"
        text += "======================================\n"

        text += f"\n{dimension} Knowledge Questions:\n"
        text += "***********************************************\n"
        for question in knowledge_questions[dimension][:limits_for_piece_of_knowledge]:
            text += f"{question}\n"

        text += f"\n{dimension} Conceptual Knowledge:\n"
        text += "***********************************************\n"
        for question in conceptual_knowledge[dimension][:limits_for_piece_of_knowledge]:
            text += f"{question}\n"

        text += f"\n{dimension} Reasoning Questions:\n"
        text += "***********************************************\n"
        for question in reasoning_questions[dimension][:limits_for_piece_of_knowledge]:
            text += f"{question}\n"

    text += "################################################################\n"
    text += f"End of Concept: {concept}\n"
    text += "################################################################\n"

    # print(text)

    row = {
        "text": text,
        "label": "everything_okay"
    }
    results.append(row)


# Save the results
output_file = "analyze_generated_knowledge/07_formatted_vlm_reasoning_benchmark.csv"
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Saved the formatted benchmark to {output_file}")