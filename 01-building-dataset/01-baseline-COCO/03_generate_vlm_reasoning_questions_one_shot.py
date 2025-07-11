# main.py

import torch
import os
import sys
import gc
import json
import pandas as pd

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.prompt.prompt_manager import PromptTemplateManager
from dotenv import load_dotenv
from libs.utils import extract_json_object

# Load environment variables
load_dotenv()
model_path = os.getenv("DEEP_SEEK_R1_32B")

# Load model
# llm = LLMWrapper(model_path=model_path)
llm = GPTLLMWrapper("gpt-4.1")

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# Load generated ontological knowledge
df = pd.read_csv("results/02_ontological_knowledge_one_shot.csv")

# Output file for step 3.
csv_filename = "results/03_generate_vlm_reasoning_questions.csv"

generated_objects = []
if os.path.exists(csv_filename):
    df_reasoning_questions = pd.read_csv(csv_filename)
    generated_objects = df_reasoning_questions['class'].to_list()
else:
    pd.DataFrame(columns=["class", "knowledge_questions", "generated_knowledge", "vlm_reasoning_questions"]).to_csv(csv_filename, index=False)

# Prepare output storage
answers = []

for _, row in df.iterrows():
    class_name = row["class"]
    knowledge_questions = json.loads(row["knowledge_questions"])
    generated_knowledge = json.loads(row["generated_knowledge"])

    if class_name in generated_objects:
        continue
    else:
        generated_objects.append(class_name)

    print("🔄 Generating Reasoning Questions for:", class_name)
    
    prompt = manager.format("03_vlm_reasoning_questions_one_shot", class_name=class_name, generated_knowledge=json.dumps(generated_knowledge, indent=2))

    response = llm(prompt, max_new_tokens=600)
    print("LLM Response", response)

    vlm_reasoning_questions = extract_json_object(response)

    print("Generated VLM Reasoning Questions", vlm_reasoning_questions)

    # Convert to DataFrame and append to CSV
    row_df = pd.DataFrame([{
        "class": class_name,
        "knowledge_questions": json.dumps(knowledge_questions),
        "generated_knowledge": json.dumps(generated_knowledge),
        "vlm_reasoning_questions": json.dumps(vlm_reasoning_questions),
    }])
    row_df.to_csv(csv_filename, mode='a', header=False, index=False)

    print(f"✅ Saved: {class_name}")

# Cleanup
# del llm
torch.cuda.empty_cache()
gc.collect()

sys.exit(0)