# main.py

import torch
import os
import sys
import gc
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.prompt.prompt_manager import PromptTemplateManager
from dotenv import load_dotenv
from utils import extract_list
from libs.coco_loader.coco_dataset_loader import COCOLoader

# Load environment variables
load_dotenv()
# model_path = os.getenv("DEEP_SEEK_R1_32B")

# Load model
llm = GPTLLMWrapper("gpt-4.1")

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# Load previously saved questions
df = pd.read_csv("results/01_ontological_queries.csv")

# Output file setup
output_filename = "results/02_ontological_knowledge_one_shot.csv"

# Write header once
generated_objects = []
if os.path.exists(output_filename):
    df_onto_queries = pd.read_csv(output_filename)
    generated_objects = df_onto_queries['class'].to_list()
else:
    pd.DataFrame(columns=["class", "knowledge_questions", "generated_knowledge"]).to_csv(output_filename, index=False)

# Prepare output storage
answers = []

for _, row in df.iterrows():
    
    class_name = row["class"]
    knowledge_questions = json.loads(row["knowledge_questions"])

    if class_name in generated_objects:
        continue
    else:
        generated_objects.append(class_name)

    print("🔄 Generating Knowledge for:", class_name)
    
    prompt = manager.format("generate_knowledge_prompt1", class_name=class_name, questions_json=json.dumps(knowledge_questions, indent=2))

    response = llm(prompt, max_new_tokens=600)
    print("LLM Response", response)

    generated_knowledge = extract_list(response)

    print("Generated Knowledge", generated_knowledge)

    # Convert to DataFrame and append to CSV
    row_df = pd.DataFrame([{
        "class": class_name,
        "knowledge_questions": json.dumps(knowledge_questions),
        "generated_knowledge": json.dumps(generated_knowledge),
    }])
    row_df.to_csv(output_filename, mode='a', header=False, index=False)


    print(f"✅ Saved: {class_name}")

# Cleanup
del llm
torch.cuda.empty_cache()
gc.collect()

sys.exit(0)