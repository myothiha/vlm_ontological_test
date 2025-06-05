import os
import sys

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import scispacy
import spacy
from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.dataset_loader.pathvqa_dataset_loader import PathVQADatasetLoader


# Load PathVQA dataset
vqa_loader = PathVQADatasetLoader(split="train")

extractor = MedicalConceptExtractor()

unique_concepts = set()

for row in vqa_loader.sample(n=1000, seed=42):
    question = row["question"]
    answer = row["answer"]
    combined_text = f"{question} {answer}"
    concepts = extractor.extract(combined_text)
    # Add only the .text of each entity to the set
    unique_concepts.update(ent.text for ent in concepts)

# Save unique concepts to disk
with open("results/unique_concepts.txt", "w") as f:
    for concept in sorted(unique_concepts):
        f.write(concept + "\n")

print(f"Total unique concepts: {len(unique_concepts)}. Saved to unique_concepts.txt.")