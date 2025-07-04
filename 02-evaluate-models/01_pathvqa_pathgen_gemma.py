import sys
import os
import h5py
from datasets import load_dataset
import io
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

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
from libs.hugging_face.DatasetManager import HuggingFaceDatasetManager

# Load Env Variables
load_dotenv()

# Load Model
model_id = os.getenv("GEMMA3_27B_PATH")
model = LVLMWrapper(model_id)

# Load the prompt templates
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# Make sure you download the NLTK data
nltk.download('punkt')

# def compute_bleu_score(reference, candidate):
#     # Tokenize
#     reference_tokens = [nltk.word_tokenize(reference.lower())]
#     candidate_tokens = nltk.word_tokenize(candidate.lower())

#     # BLEU score calculation with smoothing
#     smoothing_fn = SmoothingFunction().method1
#     score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_fn)

#     print("BLEU Score:", score)
#     return score

# def compute_bert_score(reference, candidate):
#     P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
#     f1_score = F1[0].item()
#     print("BERTScore (F1):", round(f1_score, 4))
#     return f1_score

# compute_bleu_score("This is a cat", "Is that a cat that is eating")
# compute_bert_score("This is a cat", "Is that a cat that is eating")


# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)
dataset = load_dataset("myothiha/ontobench_path_vqa", split="train", cache_dir="/mnt/synology/myothiha/HF_CACHE")

# Output result file setup
csv_output = "results/01_gemma3_yes_no_questions_zero_shot.csv"
pd.DataFrame(columns=["index", "reasoning_question", "answer", "model_answer"]).to_csv(csv_output, index=False)

result = []
for i, row in enumerate(dataset):
    if i >= 20:
        break
    
    image = row["image"]          # PIL Image object
    question = row["question"]
    answer = row["answer"]
    # concepts = row["related_concepts"]
    multiconcept_reasoning_questions = json.loads(row["multiconcept_reasoning_questions"])

    prompt = manager.format("medical_vqa_prompt", question=question)
    model_answer = model(image=image, prompt=prompt)
    row['model_answer'] = model_answer

    # # evaluate Bilingual evaluation understudy (BLEU)
    # bleu_score = compute_bleu_score(answer, model_answer)
    # row['bleu_score'] = bleu_score

    # bert_accuracy = compute_bert_score(answer, model_answer)
    # row['bert_score'] = bert_accuracy

    print("Prompt:", prompt)
    print("Question:", question)
    print("Actual Answer:", answer)
    print("Model Answer:", model_answer)

    reasoning_questions_and_answers = dict()

    for concept, reasoning_questions in multiconcept_reasoning_questions.items():

        print("Start Localization for Concept:", concept)
        
        localization_prompt = manager.format("localization_prompt", concept=concept)

        model_localization_answer = model(image=image, prompt=localization_prompt).lower()

        print("Starting Reasoning for Concept:", concept)

        yes_questions = reasoning_questions.get("yes_questions", [])
        no_questions = reasoning_questions.get("no_questions", [])
        
        positive_reasoning_results = []
        positive_accuracy = 0
        negative_accuracy = 0
        for reasoning_question in yes_questions:

            prompt = manager.format("yes_no_medical_questions_zero_shot", question=reasoning_question)
            model_positive_reasoning_answer = model(image=image, prompt=prompt).lower()

            if "yes" in model_positive_reasoning_answer.lower():
                model_positive_reasoning_answer_cleaned = "yes"
            elif "no" in model_positive_reasoning_answer.lower():
                model_positive_reasoning_answer_cleaned = "no"
            else:
                model_positive_reasoning_answer_cleaned = "unknown"

            if model_positive_reasoning_answer_cleaned == "yes":
                positive_accuracy += 1

            # Prepare the row for the result
            positive_reasoning_result = {
                "reasoning_question": reasoning_question,
                "answer": "yes",
                "model_answer": model_positive_reasoning_answer_cleaned,
                "is_correct": model_positive_reasoning_answer_cleaned == "yes"
            }
            print("Positive Reasoning Question:", positive_reasoning_result)
            positive_reasoning_results.append(positive_reasoning_result)

        negative_reasoning_results = []
        for reasoning_question in no_questions:
            prompt = manager.format("yes_no_medical_questions_zero_shot", question=reasoning_question)
            model_negative_reasoning_answer = model(image=image, prompt=prompt).lower()

            if "yes" in model_negative_reasoning_answer.lower():
                model_negative_reasoning_answer_cleaned = "yes"
            elif "no" in model_negative_reasoning_answer.lower():
                model_negative_reasoning_answer_cleaned = "no"
            else:
                model_negative_reasoning_answer_cleaned = "unknown"

            if model_negative_reasoning_answer_cleaned == "no":
                negative_accuracy += 1

            # Prepare the row for the result
            negative_reasoning_result = {
                "reasoning_question": reasoning_question,
                "answer": "no",
                "model_answer": model_negative_reasoning_answer_cleaned,
                "is_correct": model_negative_reasoning_answer_cleaned == "no"
            }
            print("Negative Reasoning Question:", reasoning_question)
            negative_reasoning_results.append(negative_reasoning_result)

        reasoning_questions_and_answers[concept] = {
            "positive_accuracy": round(positive_accuracy / len(yes_questions) if yes_questions else 0, 2),
            "negative_accuracy": round(negative_accuracy / len(no_questions) if no_questions else 0, 2),
            "positive_reasoning_results": positive_reasoning_results,
            "negative_reasoning_results": negative_reasoning_results
        }
    
    row['reasoning_result'] = json.dumps(reasoning_questions_and_answers)
    row['localization_answer'] = model_localization_answer

    # Save the result
    result.append(row)

hf_DatasetManager = HuggingFaceDatasetManager(data=result, repo_name="myothiha/ontobench_path_vqa_result", hf_token=hf_token)
hf_DatasetManager.push()
