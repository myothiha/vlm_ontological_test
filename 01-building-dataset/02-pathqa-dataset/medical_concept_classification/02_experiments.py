import os
import sys
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.concept_extractor.medical_concept_classifier import MedicalConceptClassifier

models = [
    {"simple_name": "gpt_oss_20b", "name": "gpt-oss:20b"},
    {"simple_name": "qwen3_32b", "name": "qwen3:32b"},
    {"simple_name": "openbiollm_llama3_70b", "name": "taozhiyuai/openbiollm-llama-3:70b_q2_k"},
    {"simple_name": "meditron_70b", "name": "meditron:70b"}
]

# input
dataset = "dataset/medical_concepts.csv"
concepts = pd.read_csv(dataset)

print("Dataset loaded successfully. Dataset size:", concepts.shape[0])

# experiment outputs
experiment_result = "results/experiment_result.csv"

# if result file exist, remove it.
if os.path.exists(experiment_result):
    os.remove(experiment_result)

# create empty csv file without any column header.
# pd.DataFrame(columns=["model", "experiment_result"]).to_csv(experiment_result, index=False)

no_of_iterations_for_each_model = 3

for model in models:

    model_name = model["name"]
    model_simple_name = model["simple_name"]

    llm = OllamaWrapper(model=model_name)
    classifier = MedicalConceptClassifier(llm)

    for i in range(no_of_iterations_for_each_model):
        result_file = f"results/{model_simple_name}_{i+1}.csv"

        processed_concepts = []
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            processed_concepts = df['concept'].tolist()
        else:
            df = pd.DataFrame(columns=["concept", "actual", "prediction"])

        for index, row in concepts.iterrows():
            concept = row['concept']
            actual_class = row['label']

            if concept in processed_concepts:
                continue

            predicted_class = int(classifier.classify(concept))

            row_data = {
                "concept": concept,
                "actual": actual_class,
                "prediction": predicted_class
            }

            pd.DataFrame([row_data]).to_csv(result_file, mode='a', header=not os.path.exists(result_file), index=False)

        # record experiment result
        result_df = pd.read_csv(result_file)
        actual_classes = result_df['actual'].tolist()
        predicted_classes = result_df['prediction'].tolist()

        cm = confusion_matrix(actual_classes, predicted_classes, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print("True Negative:", tn)
        print("False Positive:", fp)
        print("False Negative:", fn)
        print("True Positive:", tp)

        # Compute classification report as a dict
        report = classification_report(
            actual_classes,
            predicted_classes,
            target_names=["non-medical", "medical"],
            output_dict=True,
            digits=4
        )
        print(classification_report(
            actual_classes,
            predicted_classes,
            target_names=["non-medical", "medical"],
            output_dict=False,
            digits=4
        ))

        # Flatten into column:value dictionary
        flat_dict = {}

        for key, value in report.items():
            key = key.replace(" ", "_")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_dict[f"{key}_{subkey}"] = subvalue
            else:
                flat_dict[key] = value

        experiment_result_row = {
            "model": model_simple_name,
            # "experiment_result": json.dumps(flat_dict),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            **flat_dict,
        }

        pd.DataFrame([experiment_result_row]).to_csv(experiment_result, mode='a', header=not os.path.exists(experiment_result), index=False)