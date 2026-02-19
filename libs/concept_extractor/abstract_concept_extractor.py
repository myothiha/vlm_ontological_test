from abc import ABC, abstractmethod
import re

class AbstractConceptExtractor(ABC):
    @abstractmethod
    def extract(self, text) -> list:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __call__(self, text):
        return self.extract(text)
    
    def clean_special_chars(self, text, replacement=" "):
        """
        Replace special characters in a string with a given replacement.
        Keeps letters, numbers, and spaces by default.
        """
        if not isinstance(text, str):
            return text
        # Replace any character that is not a-z, A-Z, 0-9, or whitespace
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", replacement, text)
        
        # Collapse multiple spaces into one
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    
    def clean_concepts(self, concepts):
        concepts = {self.clean_special_chars(concept) for concept in concepts}
        concepts = {concept.strip().replace(" ", "_") for concept in concepts}
        concepts = list(sorted(concepts))
        return concepts