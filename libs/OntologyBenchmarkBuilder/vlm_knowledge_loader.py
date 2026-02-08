
import pandas as pd
import json


class VLMKnowledgeLoader:
    def __init__(self, knowledge_file):
        self._load(knowledge_file)

    def _load(self, knowledge_file):
        self.df = pd.read_csv(knowledge_file).set_index('class')
        self.conceptual_knowledge = self.df.to_dict(orient="index")

    def get_knowledge(self, concept):
        if concept in self.conceptual_knowledge:
            return json.loads(self.conceptual_knowledge[concept]['generated_knowledge'])
        else:
            return None