import os
import sys
import json
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.Evaluation.qa_evaluation_with_llm import QAEvaluationLLM
from libs.hugging_face.DatasetManager import HuggingFaceDatasetManager
from libs.OntologyBenchmarkBuilder.vlm_knowledge_loader import VLMKnowledgeLoader
from libs.OntologyBenchmarkBuilder.pipeline import OntoBenchPipeline
from libs.prompt.prompt_manager import PromptTemplateManager

# Load environment variables
load_dotenv()

# load Benchmark Data
# Paste your token here (or load from environment variable)
hf_token = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_token)

models = {
    "llava_34b": OllamaWrapper(model="llava:34b"),
    "gemma3_27b": OllamaWrapper(model="gemma3:27b-it-q4_K_M"),
    "llava_med_v1.5_mistral_7b": OllamaWrapper(model="z-uo/llava-med-v1.5-mistral-7b_f32:latest"),
    "mistral_small3.2_24b": OllamaWrapper(model="mistral-small3.2:24b"),
    "qwen2.5vl_72b": OllamaWrapper(model="qwen2.5vl:72b"),
    "llama3.2_vision_11b_instruct": OllamaWrapper(model="llama3.2-vision:11b-instruct-fp16"),
}

# Load the VLM knowledge dataset
vlm_knowledge_csv = "/home/dice/myothiha/thesis/01-building-dataset/02-pathqa-dataset/results/03_generate_vlm_reasoning_questions.csv"
knowledgeBase = VLMKnowledgeLoader(
    vlm_knowledge_csv, 
)

knowledge_about_abdomen = knowledgeBase.get_knowledge('abdomen')
# print("Loaded Knowledge:", knowledge_about_abdomen)

# Load the prompt templates
manager = PromptTemplateManager(prompt_dir="prompt_templates")

for model_name, model in models.items():

    print(f"Generate Knowledge Answers for {model_name}")

    dataset_path = f"myothiha/conceptbench_path_vqa_result_2_{model_name}_evaluated"
    dataset = load_dataset(dataset_path, split="train", cache_dir="/mnt/synology/myothiha/HF_CACHE")

    results = []
    for i, row in enumerate(dataset):
        image = row["image"]          # PIL Image object
        question = row["question"]
        all_concepts = row["extracted_concepts"]

        question_related_concepts = [concept for concept in all_concepts if concept in question]

        if question_related_concepts:
            # print(f"Question: {question}")
            # print("Related Concepts:", question_related_concepts)
            
            conceptual_knowledge = ""
            for concept in question_related_concepts:
                conceptual_knowledge += f"Knowledge about {concept}:\n"
                conceptual_knowledge += json.dumps(knowledgeBase.get_knowledge(concept), indent=2) + "\n"
            
            # print("Conceptual Knowledge:\n", conceptual_knowledge)
            
            prompt = manager.format("knowledge_ICL_medical_vqa_prompt", question=question, knowledge=conceptual_knowledge)
            knowledge_answer = model(images=[image], prompt=prompt, max_new_tokens=100)
            # print("Prompt:\n", prompt)
            print("Question:", question)
            print("Knowledge Answer:", knowledge_answer)
        else:
            knowledge_answer = row["model_answer"]

        row["knowledge_answer"] = knowledge_answer
        results.append(row)

    dataset_path += "_ICL"

    hf_DatasetManager = HuggingFaceDatasetManager(data=results, repo_name=dataset_path, hf_token=hf_token)
    hf_DatasetManager.push()
