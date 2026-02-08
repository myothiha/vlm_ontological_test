import sys
import os
import h5py
from datasets import load_dataset, Dataset
import io
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import time
from statistics import mean 

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
from datasets import load_dataset
from huggingface_hub import login

from dotenv import load_dotenv
from libs.llm_loader.llm_wrapper.lvlm_wrapper import LVLMWrapper
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.prompt.prompt_manager import PromptTemplateManager
from libs.hugging_face.DatasetManager import HuggingFaceDatasetManager

# Load Env Variables
load_dotenv()

# Load Model
# model_id = os.getenv("LLaVA")
# model = LVLMWrapper(model_id)
models = {
    "llava_34b": OllamaWrapper(model="llava:34b"),
    "gemma3_27b": OllamaWrapper(model="gemma3:27b-it-q4_K_M"),
    "llava_med_v1.5_mistral_7b": OllamaWrapper(model="z-uo/llava-med-v1.5-mistral-7b_f32:latest"),
    "mistral_small3.2_24b": OllamaWrapper(model="mistral-small3.2:24b"),
    "qwen2.5vl_72b": OllamaWrapper(model="qwen2.5vl:72b"),
    "llama3.2_vision_11b_instruct": OllamaWrapper(model="llama3.2-vision:11b-instruct-fp16"),
}

# Load the prompt templates
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)
cache_dir = "/mnt/synology/myothiha/HF_CACHE"
dataset = load_dataset("myothiha/conceptbench_path_vqa", split="train", cache_dir=cache_dir)

# Output result file setup
csv_output = "results/01_gemma3_yes_no_questions_zero_shot.csv"
pd.DataFrame(columns=["index", "reasoning_question", "answer", "model_answer"]).to_csv(csv_output, index=False)

# evaluation time result
evaluation_result_file = "results/evaluation_time.csv"

pd.DataFrame(columns=["model_name", "processed_time_in_hour"]).to_csv(evaluation_result_file, index=False)

for model_name, model in models.items():

    start = time.time()
    
    print(f"Evaluating model: {model_name}")
    result = []
    hf_output_file = f"myothiha/conceptbench_path_vqa_result_2_{model_name}"

    for i, row in enumerate(dataset):        
        image = row["image"]          # PIL Image object
        questions_and_answers = row["questions_and_answers"]
        concepts = row["all_concepts"]
        multiconcept_reasoning_questions = json.loads(row["multiconcept_reasoning_questions"])

        reasoning_questions_and_answers = dict()
        
        for concept, reasoning_questions in multiconcept_reasoning_questions.items():
            # if concept in result_cache:
            #     print("Using cached result for concept:", concept)
            #     overall_reasoning_result = result_cache[concept]
            # else:
            print("Start Localization for Concept:", concept)
            
            localization_prompt = manager.format("localization_prompt", concept=concept)
            
            print("Image Type:", type(image))
            model_localization_answer = model(images=[image], prompt=localization_prompt).lower()

            print("Localization:", model_localization_answer)

            print("Starting Reasoning for Concept:", concept)

            yes_questions = reasoning_questions.get("yes_questions", {})
            no_questions = reasoning_questions.get("no_questions", {})
            
            positive_reasoning_results = {}
            positive_accuracy = {}
            negative_accuracy = {}
            
            for dimension, questions in yes_questions.items():

                if dimension not in positive_reasoning_results.keys():
                    positive_reasoning_results[dimension] = []

                if dimension not in positive_accuracy.keys():
                    positive_accuracy[dimension] = []

                for reasoning_question in questions:
                    prompt = manager.format("yes_no_medical_questions_zero_shot", question=reasoning_question)
                    model_positive_reasoning_answer = model(images=[image], prompt=prompt).lower()

                    if "yes" in model_positive_reasoning_answer.lower():
                        model_positive_reasoning_answer_cleaned = "yes"
                    elif "no" in model_positive_reasoning_answer.lower():
                        model_positive_reasoning_answer_cleaned = "no"
                    else:
                        model_positive_reasoning_answer_cleaned = "unknown"

                    if model_positive_reasoning_answer_cleaned == "yes":
                        positive_accuracy[dimension].append(1)
                    else:
                        positive_accuracy[dimension].append(0)

                    # Prepare the row for the result
                    positive_reasoning_result = {
                        "reasoning_question": reasoning_question,
                        "answer": "yes",
                        "model_answer": model_positive_reasoning_answer_cleaned,
                        "is_correct": model_positive_reasoning_answer_cleaned == "yes"
                    }
                    print("Positive Reasoning Question:", positive_reasoning_result)
                    positive_reasoning_results[dimension].append(positive_reasoning_result)

            negative_reasoning_results = {}
            for dimension, questions in no_questions.items():

                if dimension not in negative_reasoning_results.keys():
                    negative_reasoning_results[dimension] = []

                if dimension not in negative_accuracy.keys():
                    negative_accuracy[dimension] = []

                if len(questions) > 10:
                    questions = questions[:10]

                for reasoning_question in questions:
                    prompt = manager.format("yes_no_medical_questions_zero_shot", question=reasoning_question)
                    model_negative_reasoning_answer = model(images=[image], prompt=prompt).lower()

                    if "yes" in model_negative_reasoning_answer.lower():
                        model_negative_reasoning_answer_cleaned = "yes"
                    elif "no" in model_negative_reasoning_answer.lower():
                        model_negative_reasoning_answer_cleaned = "no"
                    else:
                        model_negative_reasoning_answer_cleaned = "unknown"

                    if model_negative_reasoning_answer_cleaned == "no":
                        negative_accuracy[dimension].append(1)
                    else:
                        negative_accuracy[dimension].append(0)

                    # Prepare the row for the result
                    negative_reasoning_result = {
                        "reasoning_question": reasoning_question,
                        "answer": "no",
                        "model_answer": model_negative_reasoning_answer_cleaned,
                        "is_correct": model_negative_reasoning_answer_cleaned == "no"
                    }
                    print("Negative Reasoning Question:", reasoning_question)
                    negative_reasoning_results[dimension].append(negative_reasoning_result)

            # positive_accuracy_for_each_dimension = dict()
            # for positive_dimension, acc_list in positive_accuracy.items():
            #     if acc_list:
            #         positive_accuracy_for_each_dimension[positive_dimension] = round(mean(acc_list), 2)

            # negative_accuracy_for_each_dimension = dict()
            # for negative_dimension, acc_list in negative_accuracy.items():
            #     if acc_list:
            #         negative_accuracy_for_each_dimension[negative_dimension] = round(mean(acc_list), 2)

            positive_accuracy_for_each_dimension = {
                dim: round(mean(acc_list), 2)
                for dim, acc_list in positive_accuracy.items()
                if acc_list
            }

            negative_accuracy_for_each_dimension = {
                dim: round(mean(acc_list), 2)
                for dim, acc_list in negative_accuracy.items()
                if acc_list
            }
            # compute average accuracy            
            avg_positive_accuracy = mean(positive_accuracy_for_each_dimension.values())
            avg_negative_accuracy = mean(negative_accuracy_for_each_dimension.values())

            overall_reasoning_result = {
                "localization": model_localization_answer,
                "positive_accuracy": round(avg_positive_accuracy, 2),
                "negative_accuracy": round(avg_negative_accuracy, 2),
                "positive_accuracy_details": positive_accuracy_for_each_dimension,
                "negative_accuracy_details": negative_accuracy_for_each_dimension,
                "positive_reasoning_results": positive_reasoning_results,
                "negative_reasoning_results": negative_reasoning_results
            }
            
            # result_cache[concept] = overall_reasoning_result

            reasoning_questions_and_answers[concept] = overall_reasoning_result

        for qa in questions_and_answers:
            question = qa["question"]
            answer = qa["answer"]
            extracted_concepts = qa["extracted_concepts"]
            prompt = manager.format("medical_vqa_prompt", question=question)
            model_answer = model(images=[image], prompt=prompt)
            qa['model_answer'] = model_answer

            current_reasoning_result = dict()
            for concept in extracted_concepts:
                current_reasoning_result[concept] = reasoning_questions_and_answers.get(concept, {})

            row['reasoning_result'] = json.dumps(current_reasoning_result)

            # Save the result
            new_row = {
                "image": image,
                "question": question,
                "answer": answer,
                "model_answer": model_answer,
                "extracted_concepts": extracted_concepts,
                "reasoning_result": json.dumps(current_reasoning_result),
            }
            result.append(new_row)
        # break
    
    end = time.time()
    processed_time = end - start
    processed_time_in_hour = round(processed_time / 3600, 2)

    pd.DataFrame([[model_name, processed_time_in_hour]], columns=["model_name", "processed_time_in_hour"]).to_csv(evaluation_result_file, mode='a', header=False, index=False)

    ds = Dataset.from_list(result)
    ds.push_to_hub(
        hf_output_file,  # creates the dataset repo if it doesn't exist
        private=False
    )
    # break

# print("Reasoning result:\n", reasoning_questions_and_answers)

# hf_DatasetManager = HuggingFaceDatasetManager(data=result, repo_name="myothiha/ontobench_path_vqa_result", hf_token=hf_token)
# hf_DatasetManager.push()
