import json
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.utils import extract_list_from_ollama_models
input_json = "dataset/all.jsonl"
output_csv = "dataset/medical_concepts.csv"

# Extract all concepts from each row in the annotation file
raw_concepts = []
with open(input_json, 'r') as file:
    for line in file:
        try:
            data = json.loads(line)
            if 'label' in data:
                for label in data['label']:
                    if len(label) >= 3 and label[2] == "CONCEPT":
                        concept = data['text'][label[0]:label[1]]
                        raw_concepts.append(concept.lower())
        except Exception:
            continue

# Remove duplicates
concepts = set()
for concept in raw_concepts:
    if concept not in concepts:
        concepts.add(concept)

# Convert back to list for further processing
concepts = list(concepts)

print(f"Extracted {len(raw_concepts)} raw concepts")
print(f"Found {len(concepts)} unique concepts")

number_of_positive_label = len(concepts)

llm = OllamaWrapper(model="qwen3:32b")

prompt = f"""
You are a helpful labeler. Your task is to generate a list of concepts that are clearly non-medical and unrelated to medicine, healthcare, or biology. 
They can be anything such as daily life, Engineering, Architecture, Education, Law, Finance, Agriculture, Transportation, Art, Music, Literature, Sports, Computer Science, Programming, Culinary Arts, Fashion, Tourism, Business, Marketing, Astronomy, Physics, Chemistry, Mathematics, History, Geography, Environmental Science, Social Science, Philosophy, Politics or Media but must have no connection to medical terminology or topics. 
The generated concepts can also be a phrase like "debugging a software", "one story house".
Output your answer as a JSON array, for example: ["car", "house", "working", "eating", "programming"]
Do not provide any explanation other than a JSON array.
"""

unique_non_medical_concepts = set()

non_medical_concepts_txt = "dataset/non_medical_concepts.txt"
if os.path.exists(non_medical_concepts_txt):
    # Load medical concept list from results/unique_concepts.txt
    with open(non_medical_concepts_txt, "r") as f:
        non_medical_concepts = [line.strip() for line in f.readlines() if line.strip()]
        unique_non_medical_concepts = set(non_medical_concepts)

while len(unique_non_medical_concepts) < number_of_positive_label:
    response = llm(prompt=prompt, temperature=1)

    print("Response:", response)

    non_medical_concepts = extract_list_from_ollama_models(response)
    unique_non_medical_concepts.update(non_medical_concepts)

    # save to text file
    with open(non_medical_concepts_txt, "w") as f:
        f.writelines("\n".join(unique_non_medical_concepts))

    print("Extracted non-medical concepts:", len(unique_non_medical_concepts))

# Convert back to list for further processing
unique_non_medical_concepts = list(unique_non_medical_concepts)[:number_of_positive_label]

print("Number of unique non-medical concepts:", len(unique_non_medical_concepts))

# Create DataFrame and write to CSV for binary classification (label=1)
df_medical = pd.DataFrame({'concept': concepts, 'label': [1]*len(concepts)})
df_non_medical = pd.DataFrame({'concept': unique_non_medical_concepts, 'label': [0]*len(unique_non_medical_concepts)})
df = pd.concat([df_medical, df_non_medical], ignore_index=True)

df.to_csv(output_csv, index=False)

print(f"Created {output_csv} file with binary classification labels")