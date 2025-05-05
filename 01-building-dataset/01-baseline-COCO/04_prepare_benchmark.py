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
from datasets import Dataset, Features, Value, Array3D, Image as HFImage

# Load environment variables
load_dotenv()

# Load category list from COCO
loader = COCOLoader()
categories = loader.get_all_categories()

# Input file
csv_input = "results/03_generate_vlm_reasoning_questions.csv"

# Output file setup
csv_output = "results/04_benchmark_COCO.csv"

# Write header once
pd.DataFrame(columns=["image_url", "class_name", "reasoning_question", "answer"]).to_csv(csv_output, index=False)

# Load reasoning questions
df = pd.read_csv(csv_input).set_index('class')
reasoning_questions = df.to_dict(orient="index")
# print(reasoning_questions)

image_ids = loader.get_image_ids()
print("Total Images: ", len(image_ids))
no_of_processed_images = 1
min_area = 10000
for image_id in image_ids:
    image, image_info = loader.load_image(image_id)
    image_url = image_info['coco_url']
    annotations = loader.get_annotations(image_id)
    categories = loader.get_detected_objects(image_id)
    questions = []

    # This will store object-class and bbox for each object in the image
    objects = []
    output_dir = "./cropped_image"
    for ann in annotations:
        category_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = map(int, bbox)
        category_name = loader.coco.loadCats(category_id)[0]["name"]

        # Prepare output path
        save_dir = os.path.join(output_dir, category_name)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{image_id}_{ann['id']}.jpg"
        save_path = os.path.join(save_dir, filename)

        # Save cropped image
        cropped.save(save_path)

        objects.append({
            "class_name": category_name,
            "image": cropped
        })
    
    print(objects)

    for class_name in categories:
        
        break
        
        try:
            vlm_reasoning_questions = json.loads(reasoning_questions[class_name]['vlm_reasoning_questions'])
        except:
            # if there is no related info for a given class skip it.
            continue
        
        positive_questions = vlm_reasoning_questions['yes_questions']
        negative_questions = vlm_reasoning_questions['no_questions']

        for p_question in positive_questions:
            row_df = pd.DataFrame([{
                "image_url": image_url,
                "class_name": class_name,
                "reasoning_question": p_question,
                "answer": 'yes'
            }])

            row_df.to_csv(csv_output, mode='a', header=False, index=False)

        for n_question in negative_questions:
            row_df = pd.DataFrame([{
                "image_url": image_url,
                "class_name": class_name,
                "reasoning_question": n_question,
                "answer": 'no'
            }])

            row_df.to_csv(csv_output, mode='a', header=False, index=False)
    break

# for image_id in image_ids:
#     image, image_info = loader.load_image(image_id)
#     image_url = image_info['coco_url']
#     annotations = loader.get_annotations(image_id)
#     categories = loader.get_detected_objects(image_id)
#     questions = []

#     no_of_negative_questions = 0
#     for class_name, data in reasoning_questions.items():
#         questions = json.loads(data['vlm_reasoning_questions'])
#         if class_name in categories:
#             answer = "yes"
#         else:
#             answer = "no"
#             no_of_negative_questions += 1
#             if no_of_negative_questions > 50:
#                 continue

#         for question in questions:
#             # Convert to DataFrame and append to CSV
#             row_df = pd.DataFrame([{
#                 "image_url": image_url,
#                 "class_name": class_name,
#                 "reasoning_question": question,
#                 "answer": answer
#             }])

#             row_df.to_csv(csv_output, mode='a', header=False, index=False)

#     print(f"Processed {no_of_processed_images} Images")
#     no_of_processed_images += 1