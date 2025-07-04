from libs.concept_extractor.abstract_concept_extractor import AbstractConceptExtractor
from pathlib import Path
import os
from libs.prompt.prompt_manager import PromptTemplateManager

class LLMBasedMedicalConceptExtractor(AbstractConceptExtractor):
    def __init__(self, model):
        self.llm = model

        self.current_dir = Path(__file__).resolve().parent
        prompt_dir = os.path.join(self.current_dir, "prompt_template")

        self.prompt_manager = PromptTemplateManager(prompt_dir=prompt_dir)

    def extract(self, text) -> list:
        """
        Extracts medical concepts from the given text using the LLM model.
        
        :param text: The input text from which to extract medical concepts.
        :return: A list of extracted medical concepts.
        """
        # This method should implement the logic to use the LLM model to extract concepts
        # For now, we return an empty list as a placeholder

        prompt = self.prompt_manager.format("extract_medical_concepts", text=text)
        response = self.llm()

        return []
    
    def extractUnique(self, text) -> list:
        """
        Extracts unique medical concepts from the given text using the LLM model.
        
        :param text: The input text from which to extract unique medical concepts.
        :return: A list of unique extracted medical concepts.
        """
        return list(set(self.extract(text)))
    
    def extract_from_llm(response) -> list:
        """
        Extracts medical concepts from the LLM response.
        
        :param response: The response from the LLM containing medical concepts.
        :return: A list of extracted medical concepts.
        """
        # This method should implement the logic to parse the LLM response
        # For now, we return an empty list as a placeholder
        return []