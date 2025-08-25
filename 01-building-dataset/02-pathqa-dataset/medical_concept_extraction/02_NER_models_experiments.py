import os
import sys
import json
import pandas as pd
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.NER_Evaluation.evaluation_methods import relax_evaluation
models = [
    {"simple_name": "gpt_oss_20b", "name": "gpt-oss:20b"},
    {"simple_name": "qwen3_32b", "name": "qwen3:32b"},
    {"simple_name": "openbiollm_llama3_70b", "name": "taozhiyuai/openbiollm-llama-3:70b_q2_k"},
    {"simple_name": "meditron_70b", "name": "meditron:70b"}
]

# input
dataset = "dataset/medical_ner_dataset.csv"
concepts = pd.read_csv(dataset)

print("Dataset loaded successfully. Dataset size:", concepts.shape[0])

# experiment outputs
experiment_result = "results/experiment_result.csv"
pd.DataFrame(columns=["model", "precision", "recall", "f1"]).to_csv(experiment_result, index=False)

no_of_iterations_for_each_model = 3

for model in models:

    model_name = model["name"]
    model_simple_name = model["simple_name"]

    llm = OllamaWrapper(model=model_name)
    concept_extractor = LLMBasedMedicalConceptExtractor(model=llm)

    for i in range(no_of_iterations_for_each_model):
        result_file = f"results/{model_simple_name}_{i+1}.csv"

        processed_data_ids = []
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            processed_data_ids = df['id'].tolist()
        else:
            df = pd.DataFrame(columns=["id", "concept", "actual", "prediction"])

        for index, row in concepts.iterrows():
            id = row['id']
            text = row['text']
            actual_concepts = json.loads(row['concepts'])

            if id in processed_data_ids:
                continue
            
            extracted_concepts = concept_extractor.extract(text)

            evaluations = relax_evaluation(actual_concepts, extracted_concepts)

            row_data = {
                "id": id,
                "actual_concepts": actual_concepts,
                "extracted_concepts": extracted_concepts,
                "true_positive": evaluations['true_positives'],
                "false_positive": evaluations['false_positives'],
                "false_negative": evaluations['false_negatives'],
            }

            pd.DataFrame([row_data]).to_csv(result_file, mode='a', header=not os.path.exists(result_file), index=False)
        
        # record experiment result
        result_df = pd.read_csv(result_file)
        true_positives = result_df['true_positive'].sum()
        false_positives = result_df['false_positive'].sum()
        false_negatives = result_df['false_negative'].sum()

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        experiment_result_row = {
            "model": model_simple_name,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1_score), 4)
        }

        print(experiment_result_row)

        pd.DataFrame([experiment_result_row]).to_csv(experiment_result, mode='a', header=not os.path.exists(experiment_result), index=False)