# main.py

import torch
import os
import sys

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gc
import json
import pandas as pd

from dotenv import load_dotenv
from libs.coco_loader.coco_dataset_loader import COCOLoader

from PIL import Image

from datasets import load_dataset, Dataset, Features, Value, Image
from huggingface_hub import login
import shutil


# Load environment variables
load_dotenv()

# Load category list from COCO
loader = COCOLoader()
categories = loader.get_all_categories()

# Input file
csv_input = "results/03_generate_vlm_reasoning_questions.csv"

# Output file setup
csv_output = "results/04_benchmark_COCO.csv"
image_output_dir = "results/images"
os.makedirs(image_output_dir, exist_ok=True)

# Write CSV header (only once)
pd.DataFrame(columns=["image_path", "class_name", "question", "answer"]).to_csv(csv_output, index=False)

# Load reasoning questions
df = pd.read_csv(csv_input).set_index('class')
reasoning_questions = df.to_dict(orient="index")

image_ids = loader.get_image_ids()
print("Total Images: ", len(image_ids))
image_counter = 0
min_area = 4096

for image_id in image_ids:
    image, image_info = loader.load_image(image_id)
    annotations = loader.get_annotations(image_id)

    for ann in annotations:
        category_id = ann["category_id"]
        category_name = loader.coco.loadCats(category_id)[0]["name"]

        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Ensure crop doesn't exceed image bounds
        img_width, img_height = image.size
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, img_width)
        y2 = min(y + h, img_height)

        if x2 > x1 and y2 > y1:
            cropped = image.crop((x1, y1, x2, y2))
        else:
            continue  # Skip invalid crop

        # Crop using PIL
        cropped = image.crop((x, y, x + w, y + h))
        
        width, height = cropped.size
        area = width * height

        if area < min_area:
            # skip the image
            print(f"Skipped: {width}x{height} = {area} px²")
            continue

        # Save image to file
        image_filename = f"{image_counter:08d}.png"
        image_path = os.path.join(image_output_dir, image_filename)
        cropped.save(image_path)

        try:
            vlm_reasoning_questions = json.loads(reasoning_questions[category_name]['vlm_reasoning_questions'])
        except:
            continue

        for answer_type in ["yes_questions", "no_questions"]:
            answer = "yes" if answer_type == "yes_questions" else "no"
            for question in vlm_reasoning_questions[answer_type]:

                # Append to CSV
                new_row = pd.DataFrame([{
                    "image_path": image_path,
                    "class_name": category_name,
                    "question": question,
                    "answer": answer
                }])
                new_row.to_csv(csv_output, mode="a", index=False, header=False)

                image_counter += 1


image_output_dir = "results/images"

# STEP 1: Authenticate
login(token=os.getenv("HF_ACCESS_TOKEN"))  # Or just `login()` to log in interactively

# STEP 2: Copy files to a temporary repo folder
dataset_repo_dir = "ontobench_coco_repo"
os.makedirs(dataset_repo_dir, exist_ok=True)

# Copy CSV and image folder
shutil.copy("results/04_benchmark_COCO.csv", os.path.join(dataset_repo_dir, "data.csv"))
shutil.copytree("results/images", os.path.join(dataset_repo_dir, "images"), dirs_exist_ok=True)

# STEP 3: Load dataset from CSV (image paths relative)
features = Features({
    "image_path": Value("string"),
    "class_name": Value("string"),
    "question": Value("string"),
    "answer": Value("string")
})

dataset = load_dataset("csv", data_files=os.path.join(dataset_repo_dir, "data.csv"), features=features, split="train")

# STEP 4: Convert image_path column to Image() column
def convert_path_to_image(example):
    return {"image": example["image_path"]}

dataset = dataset.map(convert_path_to_image)
dataset = dataset.remove_columns("image_path")
dataset = dataset.cast_column("image", Image())

try:
    # STEP 5: Push to Hub
    dataset.push_to_hub("myothiha/ontobench_coco")
    print("✅ Dataset pushed successfully to Hugging Face.")

    # Clean up folders
    if os.path.exists(dataset_repo_dir):
        shutil.rmtree(dataset_repo_dir)
        print(f"🧹 Deleted temporary dataset folder: {dataset_repo_dir}")

    if os.path.exists(image_output_dir):
        shutil.rmtree(image_output_dir)
        print(f"🧹 Deleted cropped image folder: {image_output_dir}")

except Exception as e:
    print("❌ Failed to push dataset:", e)
    print("⚠️ Keeping local files for debugging.")
