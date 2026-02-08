import torch
import os
import sys

from PIL import Image
import imagehash

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from libs.OntologyBenchmarkBuilder.pipeline import OntoBenchPipeline

from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper

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

# Setup pipeline to generate questions for concepts that are not exist.
llm = GPTLLMWrapper("gpt-4.1")
# llm = OllamaWrapper(model="gpt-oss:20b")

concept_bench_pipeline = OntoBenchPipeline(
    llm = llm, # Use for negative questions generation. LLM encoder to check most dissimilar concepts. None means randomly sample.
    knowledge_question_prompt_templates = "prompt_templates/kq_prompt_templates",
    generate_knowledge_prompt_template = "02_generate_knowledge_prompt1",
    vlm_reasoning_prompt_template = "03_vlm_reasoning_questions_one_shot"
)

# load dataset
df = pd.read_csv(csv_input).set_index('class')

# Load reasoning questions
num_samples = None
if num_samples:
    df = df.sample(n=num_samples, random_state=42)


# print("Columns: ", df.columns.to_list())
dataset = df.to_dict(orient="index")

all_concepts = list(dataset.keys())
print(f"Selecting Random Concepts ({len(all_concepts)}): ")
# print(all_concepts)

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
no_empty_kq = 0
no_empty_ck = 0
no_empty_rq = 0
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
    limits_for_piece_of_knowledge = 5
    for dimension in conceptual_dimensions:
        text += f"\nDimension: {dimension}\n"
        text += "======================================\n"

        text += f"\n{dimension} Knowledge Questions:\n"
        text += "***********************************************\n"

        for question in knowledge_questions[dimension][:limits_for_piece_of_knowledge]:
            text += f"{question}\n"

        if len(knowledge_questions[dimension]) < 1:
            no_empty_kq += 1

        if len(conceptual_knowledge[dimension]) < 1:
            no_empty_ck += 1

        if len(reasoning_questions[dimension]) < 1:
            no_empty_rq += 1

        # text += f"\n{dimension} Conceptual Knowledge:\n"
        # text += "***********************************************\n"
        # for question in conceptual_knowledge[dimension][:limits_for_piece_of_knowledge]:
        #     text += f"{question}\n"

        # text += f"\n{dimension} Reasoning Questions:\n"
        # text += "***********************************************\n"
        # for question in reasoning_questions[dimension][:limits_for_piece_of_knowledge]:
        #     text += f"{question}\n"

    text += "################################################################\n"
    text += f"End of Concept: {concept}\n"
    text += "################################################################\n"

    # print(text)

    row = {
        "text": text,
        "label": "everything_okay"
    }
    results.append(row)

print("Empty knowledge questions:",  no_empty_kq)
print("Empty conceptual knowledge:",  no_empty_ck)
print("Empty reasoning questions:",  no_empty_rq)

# Save the results
output_file = "analyze_generated_knowledge/07_formatted_vlm_reasoning_benchmark.csv"
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Saved the formatted benchmark to {output_file}")