import sys
import os
from datasets import load_dataset

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json

from dotenv import load_dotenv
from libs.llm_loader.lvlm_wrapper import LVLMWrapper
from libs.coco_loader.coco_dataset_loader import COCOLoader
from libs.prompt.prompt_manager import PromptTemplateManager

# Load Env Variables
load_dotenv()

# Load Model
model_id = os.getenv("GEMMA3_27B_PATH")
model = LVLMWrapper(model_id)

# Load category list from COCO
loader = COCOLoader()

# Load the prompt templates
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# load Benchmark Data
# Load the dataset from Hugging Face Hub
dataset = load_dataset("myothiha/ontobench_coco", split="train")

# Output result file setup
csv_output = "results/01_gemma3_yes_no_questions_zero_shot.csv"
pd.DataFrame(columns=["image_url", "reasoning_question", "answer", "model_answer"]).to_csv(csv_output, index=False)

for row in dataset:
    
    image_url = row['image_url']
    reasoning_question = row['reasoning_question']
    answer = row['answer']
    prompt = prompt = manager.format("yes_no_questions_zero_shot", question=reasoning_question)
    model_answer = model(image_url = image_url,
                prompt = prompt)
    
    print("Image:", image_url)
    print("Question:", reasoning_question)
    print("Actual Answer:", answer)
    print("Model Answer:", model_answer)

    row_df = pd.DataFrame([{
        "image_url": row['image_url'],
        "reasoning_question": reasoning_question,
        "answer": answer,
        "model_answer": model_answer
    }])

    row_df.to_csv(csv_output, mode='a', header=False, index=False)
    break