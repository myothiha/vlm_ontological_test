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
from libs.prompt.prompt_manager import PromptTemplateManager
from dotenv import load_dotenv
from utils import extract_list
from libs.coco_loader.coco_dataset_loader import COCOLoader

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# Load environment variables
load_dotenv()
model_path = os.getenv("DEEP_SEEK_R1_70B")

# Load model
llm = LLMWrapper(model_path=model_path, quantized=True, quantization_bits=4)

# Load category list from COCO
loader = COCOLoader()
categories = loader.get_all_categories()

# Output file setup
csv_filename = "01_ontological_queries.csv"

# Write header once
pd.DataFrame(columns=["class", "knowledge_questions"]).to_csv(csv_filename, index=False)

# Generate and save questions per class
for class_name in categories:
    print("🔄 Generating Knowledge Questions for:", class_name)

    prompt = manager.format("generate_knowledge_questions_one_shot", class_name=class_name)
    # print("Prompt:", prompt)

    result = llm(prompt, max_new_tokens=400)
    response = result[0]['generated_text']

    # print("Response", response)

    knowledge_questions = extract_list(response)

    print("Knowledge Questions:", knowledge_questions)

    # Convert to DataFrame and append to CSV
    row_df = pd.DataFrame([{
        "class": class_name,
        "knowledge_questions": json.dumps(knowledge_questions)
    }])
    row_df.to_csv(csv_filename, mode='a', header=False, index=False)

    print(f"✅ Saved: {class_name}")

    break

# Cleanup
del llm
torch.cuda.empty_cache()
gc.collect()

sys.exit(0)