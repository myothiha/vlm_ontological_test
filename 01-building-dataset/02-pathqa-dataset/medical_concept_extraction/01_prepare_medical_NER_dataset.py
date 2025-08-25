import json
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.utils import extract_list_from_ollama_models, clean_special_chars
input_json = "dataset/all.jsonl"
output_csv = "dataset/medical_ner_dataset.csv"

# Extract all concepts from each row in the annotation file
data_rows = []
with open(input_json, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            obj = json.loads(line)
            id = obj.get("id")
            text = obj.get("text", "")
            labels = obj.get("label", [])
            concepts = set()
            for label in labels:
                if isinstance(label, list) and len(label) == 3 and label[2] == "CONCEPT":
                    start, end = label[0], label[1]
                    concept = text[start:end].strip()
                    concepts.add(concept)
            data_rows.append({
                "id": id,
                "text": text,
                "concepts": json.dumps(list(concepts), ensure_ascii=False)
            })
        except Exception as e:
            print(f"Error processing line: {e}")
            continue

df = pd.DataFrame(data_rows)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} rows to {output_csv}")
