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
from libs.dataset_loader.coco_dataset_loader import COCOLoader

# Load environment variables
load_dotenv()
model_path = os.getenv("DEEP_SEEK_R1_32B")

# Load model
llm = LLMWrapper(model_path=model_path)

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt/templates")

# Load previously saved questions
df = pd.read_csv("ontological_queries.csv")

# Output file setup
csv_filename = "ontological_queries.csv"

# Prepare output storage
answers = []

for index, row in df.iterrows():
    class_name = row["class"]

    if row["knowledge_questions"] != "repeat":
        continue
    
    print("🔄 Generating Knowledge Questions for:", class_name)
    
    prompt = manager.format("generate_knowledge_questions_one_shot", class_name=class_name)

    result = llm(prompt, max_new_tokens=400)
    response = result[0]['generated_text']

    # print("LLM Response", response)

    knowledge_questions = extract_list(response)
    print("Knowledge Questions:", json.dumps(knowledge_questions, indent=2))
    
    df.loc[index, "knowledge_questions"] = json.dumps(knowledge_questions)
    
    df.to_csv(csv_filename, index=False)

    print(f"✅ Saved: {class_name}")
    # break

# Cleanup
# del llm
torch.cuda.empty_cache()
gc.collect()

sys.exit(0)