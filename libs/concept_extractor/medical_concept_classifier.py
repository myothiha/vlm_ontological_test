import os
import json
import re
from pathlib import Path

from libs.prompt.prompt_manager import PromptTemplateManager
from libs.concept_extractor.abstract_concept_classifier import AbstractConceptClassifier

class MedicalConceptClassifier(AbstractConceptClassifier):
    def __init__(self, llm):
        self.llm = llm
        self.current_dir = Path(__file__).resolve().parent

        prompt_dir = os.path.join(self.current_dir, "prompt_template")
        self.prompt_manager = PromptTemplateManager(prompt_dir)

    def classify(self, concept: str) -> bool:
        prompt = self.prompt_manager.format("medical_concept_classification", text=concept)
        response = self.llm(prompt).lower()
        
        return "true" in response or "yes" in response
