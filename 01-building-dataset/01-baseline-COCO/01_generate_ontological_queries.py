# main.py

import torch
import os
import sys
import gc
import json
import pandas as pd

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.llm_loader.llm_wrapper import LLMWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.prompt.prompt_manager import PromptTemplateManager
from dotenv import load_dotenv
from libs.utils import extract_list
from libs.dataset_loader.coco_dataset_loader import COCOLoader

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# Load environment variables
load_dotenv()
# model_path = os.getenv("DEEP_SEEK_R1_32B")

# Load model
# llm = LLMWrapper(model_path=model_path, quantized=True, quantization_bits=4)
llm = GPTLLMWrapper("gpt-4.1")

# Load category list from COCO
loader = COCOLoader()
categories = loader.get_all_categories()

# Output file setup
csv_filename = "results/01_ontological_queries.csv"

generated_objects = []
# Write header once
if os.path.exists(csv_filename):
    df_onto_queries = pd.read_csv(csv_filename)
    generated_objects = df_onto_queries['class'].to_list()
else:
    pd.DataFrame(columns=["class", "knowledge_questions"]).to_csv(csv_filename, index=False)

# Generate and save questions per class
for class_name in categories:
    # skip if we already generated knowledge for this item.

    if class_name in generated_objects:
        continue
    else:
        generated_objects.append(class_name)

    print("🔄 Generating Knowledge Questions for:", class_name)

    prompt = manager.format("generate_knowledge_questions_one_shot", class_name=class_name)
    # print("Prompt:", prompt)

    response = llm(prompt, max_new_tokens=400)

    print("Response", response)

    knowledge_questions = extract_list(response)

    print("Knowledge Questions:", knowledge_questions)

    # Convert to DataFrame and append to CSV
    row_df = pd.DataFrame([{
        "class": class_name,
        "knowledge_questions": json.dumps(knowledge_questions)
    }])
    row_df.to_csv(csv_filename, mode='a', header=False, index=False)

    print(f"✅ Saved: {class_name}")

# Cleanup
del llm
torch.cuda.empty_cache()
gc.collect()

sys.exit(0)