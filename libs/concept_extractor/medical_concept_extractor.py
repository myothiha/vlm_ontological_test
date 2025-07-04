
import spacy
from libs.concept_extractor.abstract_concept_extractor import AbstractConceptExtractor

class MedicalConceptExtractor(AbstractConceptExtractor):
    def __init__(self, model_name="en_core_sci_scibert"):
        self.nlp = spacy.load(model_name)

    def extract(self, text) -> list:
        doc = self.nlp(text)
        return list(doc.ents)
    
    def extractUnique(self, text) -> list:
        doc = self.nlp(text)
        unique_concepts = set()
        unique_concepts.update(ent.text for ent in doc.ents)
        return list(unique_concepts)
