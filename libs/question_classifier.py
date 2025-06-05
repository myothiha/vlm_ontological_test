from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class BaseQuestionClassifier(ABC):
    @abstractmethod
    def classify(self, class_name, question):
        """
        Returns 1 if the question is related to the class, 0 otherwise.
        """
        pass

class LLMBasedQuestionClassifier(BaseQuestionClassifier):
    def __init__(self, model_dir, device=None):
        """
        model_dir: path to the folder containing the saved LLM model and tokenizer (from save_pretrained)
        device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    def classify(self, class_name, question):
        text = f"{class_name} [SEP] {question}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
        return int(pred)
