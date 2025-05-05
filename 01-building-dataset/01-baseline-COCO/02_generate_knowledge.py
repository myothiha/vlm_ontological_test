# main.py

import torch
import os
import sys
import gc
import json
import pandas as pd

from llm_loader.llm_wrapper import LLMWrapper
from prompt.prompt_manager import PromptTemplateManager
from dotenv import load_dotenv
from utils import extract_list
from coco_loader.coco_dataset_loader import COCOLoader

# Load environment variables
load_dotenv()
model_path = os.getenv("DEEP_SEEK_R1_32B")

# Load model
llm = LLMWrapper(model_path=model_path)

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt/templates")

# Load previously saved questions
df = pd.read_csv("results/01_ontological_queries.csv")

# Output file setup
csv_filename = "02_ontological_knowledge_one_shot.csv"
pd.DataFrame(columns=["class", "knowledge_questions", "generated_knowledge"]).to_csv(csv_filename, index=False)

# Prepare output storage
answers = []

for _, row in df.iterrows():
    
    class_name = row["class"]
    knowledge_questions = json.loads(row["knowledge_questions"])

    print("🔄 Generating Knowledge Questions for:", class_name)
    
    prompt = manager.format("generate_knowledge_prompt1", class_name=class_name, questions_json=json.dumps(knowledge_questions, indent=2))

    result = llm(prompt, max_new_tokens=600)
    response = result[0]['generated_text']

    # print("LLM Response", response)

    generated_knowledge = extract_list(response)

    print("Generated Knowledge", generated_knowledge)

    # Convert to DataFrame and append to CSV
    row_df = pd.DataFrame([{
        "class": class_name,
        "knowledge_questions": json.dumps(knowledge_questions),
        "generated_knowledge": json.dumps(generated_knowledge),
    }])
    row_df.to_csv(csv_filename, mode='a', header=False, index=False)


    print(f"✅ Saved: {class_name}")

# Cleanup
del llm
torch.cuda.empty_cache()
gc.collect()

sys.exit(0)