import spacy

class MedicalConceptExtractor:
    def __init__(self, model_name="en_core_sci_scibert"):
        self.nlp = spacy.load(model_name)

    def extract(self, text):
        doc = self.nlp(text)
        return list(doc.ents)
