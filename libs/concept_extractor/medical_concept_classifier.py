import os
import json
import re
from libs.prompt.prompt_manager import PromptTemplateManager

from pathlib import Path

class MedicalConceptClassifier:
    def __init__(self, llm):
        self.llm = llm
        self.current_dir = Path(__file__).resolve().parent

        prompt_dir = os.path.join(self.current_dir, "prompt_template")
        self.prompt_manager = PromptTemplateManager(prompt_dir)

    def classify(self, text):
        prompt = self.prompt_manager.format("medical_concept_classification", text=text)
        response = self.llm(prompt).lower()
        # print("Classification response:", response)
        return "true" in response or "yes" in response
