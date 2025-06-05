import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from libs.question_classifier import LLMBasedQuestionClassifier

# Instantiate the question classifier
# Path to the saved Roberta model and tokenizer
classifier_model_dir = os.path.join(os.path.dirname(__file__), "classifier/roberta")
classifier = LLMBasedQuestionClassifier(classifier_model_dir)

# Example test cases
examples = [
    ("liver", "What biological functions does the liver perform?"),
    ("heart", "What is the capital of France?"),
    ("lung", "Which organ systems include the lung?")
]

for class_name, question in examples:
    label = classifier.classify(class_name, question)
    print(f"Class: {class_name} | Question: {question}\nPredicted label: {label} ({type(label)})\n")