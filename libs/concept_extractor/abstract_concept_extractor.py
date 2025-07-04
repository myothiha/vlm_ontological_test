from abc import ABC, abstractmethod

class AbstractConceptExtractor:
    @abstractmethod
    def extract(self, text) -> list:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def extractUnique(self, text) -> list:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __call__(self, text):
        return self.extract(text)