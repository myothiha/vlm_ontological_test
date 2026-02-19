from abc import ABC, abstractmethod
import re

class AbstractConceptClassifier(ABC):
    @abstractmethod
    def classify(self, concept: str) -> bool:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __call__(self, concept: str) -> bool:
        return self.classify(concept)