import sys
import os
import h5py
from datasets import load_dataset
import io
import numpy as np

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
from datasets import load_dataset
from huggingface_hub import login

from dotenv import load_dotenv
from libs.llm_loader.llm_wrapper.lvlm_wrapper import LVLMWrapper
from libs.dataset_loader.coco_dataset_loader import COCOLoader
from libs.prompt.prompt_manager import PromptTemplateManager

# Load Env Variables
load_dotenv()

# Load Model
model_id = os.getenv("GEMMA3_27B_PATH")
model = LVLMWrapper(model_id)

# Load category list from COCO
loader = COCOLoader()
classes = loader.get_all_categories()

print("COCO Classes:", classes)

# Load the prompt templates
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)
dataset = load_dataset("myothiha/ontobench_coco", split="train", cache_dir="/mnt/synology/myothiha/HF_CACHE")

# Output result file setup
csv_output = "results/01_gemma3_yes_no_questions_zero_shot.csv"
pd.DataFrame(columns=["index", "reasoning_question", "answer", "model_answer"]).to_csv(csv_output, index=False)

# Output HDF5 file
# hdf5_output = "results/01_gemma3_yes_no_questions_zero_shot.h5"

prompt = manager.format("multi_classification", class_list=classes)

# Step 3: Extract structured list from output
def parse_detected_objects(output_text, class_list):
    output_text = output_text.lower()
    detected = [cls for cls in class_list if cls in output_text]
    return list(sorted(set(detected)))  # remove duplicates, sort for consistency

min_area = 1024 # Minimum area for annotations to be considered valid

for i, row in enumerate(dataset):
    print("Image Id:", row["image_id"])
    image = row["image"]          # PIL Image object
    annotations = loader.get_annotations(row["image_id"])

    concepts = json.loads(row["concepts"])
    print("Concepts:", concepts)
    truth_classes = []
    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Ensure crop doesn't exceed image bounds
        img_width, img_height = image.size
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, img_width)
        y2 = min(y + h, img_height)
        area = (x2 - x1) * (y2 - y1)

        if area > min_area:
            # remove small annotations
            category_name = loader.coco.loadCats(ann["category_id"])[0]["name"]
            # print(f"Add large annotation: {category_name} with area {area} > {min_area}")
            truth_classes.append(category_name)

    print("Truth Classes:", list(sorted(set(truth_classes))))

    model_answer = model(image=image, prompt=prompt)
    print("Model Answer:", model_answer)
    detected_objects = parse_detected_objects(model_answer, classes)
    print("Detected Objects:", detected_objects)

    break

    # Convert image to bytes (PNG format)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()

    # Save data in a group
    grp = f.create_group(str(i))
    grp.attrs["index"] = i
    grp.create_dataset("image", data=np.void(img_bytes))
    grp.attrs["reasoning_question"] = reasoning_question
    grp.attrs["answer"] = answer
    grp.attrs["model_answer"] = model_answer

    row_df = pd.DataFrame([{
        "index": i,
        "reasoning_question": reasoning_question,
        "answer": answer,
        "model_answer": model_answer
    }])

    row_df.to_csv(csv_output, mode='a', header=False, index=False)


# for row in dataset:
    
#     image = row["image"]          # PIL Image object
#     reasoning_question = row["question"]
#     answer = row["answer"]
#     class_name = row["class_name"]

#     prompt = prompt = manager.format("yes_no_questions_zero_shot", question=reasoning_question)
#     model_answer = model(image = image,
#                 prompt = prompt)
    
#     # print("Image:", image)
#     print("Question:", reasoning_question)
#     print("Actual Answer:", answer)
#     print("Model Answer:", model_answer)

#     row_df = pd.DataFrame([{
#         "image": image,
#         "reasoning_question": reasoning_question,
#         "answer": answer,
#         "model_answer": model_answer
#     }])

#     row_df.to_csv(csv_output, mode='a', header=False, index=False)
#     break