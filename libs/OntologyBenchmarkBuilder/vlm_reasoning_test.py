import pandas as pd
import json
import random
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VLMReasoningQuestionsLoader:
    def __init__(self, file_path, llm_model_path = None):
        self._load(file_path)
        self.llm_model_path = llm_model_path
        self.tokenizer = None
        self.model = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load(self, file_path):
        df = pd.read_csv(file_path).set_index('class')
        reasoning_questions = df.to_dict(orient="index")
        self.reasoning_questions = reasoning_questions
        return reasoning_questions
    
    def _load_language_model(self):
        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
            self.model = AutoModel.from_pretrained(self.llm_model_path)
            self.model.eval()

            self.model = self.model.to(self.device)

    def get_positive_questions(self, concept_class=None):
        """
        Returns the loaded reasoning questions.
        
        :return: Dictionary of reasoning questions indexed by class.
        """
        if not hasattr(self, 'reasoning_questions'):
            raise ValueError("Reasoning questions have not been loaded. Call load(path_to_csv) first.")
        
        # If a specific concept class is provided, filter the questions
        if concept_class:
            return json.loads(self.reasoning_questions[concept_class]['vlm_reasoning_questions'])
        return self.reasoning_questions['vlm_reasoning_questions']


    def get_negative_questions(self, concept_class=None, exclude_concepts=None, num_concepts=3, num_questions=10):
        """
        Returns negative reasoning questions: for a given concept, pick random other concepts (excluding concept_class and exclude_concepts) and get random questions from those concepts.
        :param concept_class: The concept to exclude from negative sampling.
        :param exclude_concepts: List of additional concepts to exclude from negative sampling.
        :param num_concepts: Number of random concepts to sample.
        :param num_questions: Number of random questions to sample from the selected concepts.
        :return: List of negative reasoning questions.
        """
        if not hasattr(self, 'reasoning_questions'):
            raise ValueError("Reasoning questions have not been loaded. Call load(path_to_csv) first.")
        
        available_classes = self.get_negative_concepts(concept_class=concept_class, exclude_concepts=exclude_concepts)

        if len(available_classes) < num_concepts:
            raise ValueError("Not enough other concepts with questions to sample from.")
        
        if self.llm_model_path is None:
            sampled_classes = random.sample(available_classes, num_concepts)
        else:
            if self.tokenizer is None or self.model is None:
                self._load_language_model()
            dissimilar_concepts = self.find_top_k_dissimilar(concept_class, available_classes, k=num_concepts)
            sampled_classes = [concept for concept, _ in dissimilar_concepts]
        
        print("Negative Concepts", sampled_classes)

        negative_questions = []
        for c in sampled_classes:
            questions = self.reasoning_questions[c]["vlm_reasoning_questions"]
            # If questions is a stringified list, parse it
            if isinstance(questions, dict) and 'questions' in questions:
                qlist = questions['questions']
            elif isinstance(questions, str):
                try:
                    qlist = json.loads(questions)
                except Exception:
                    qlist = []
            else:
                qlist = questions
            if isinstance(qlist, str):
                try:
                    qlist = json.loads(qlist)
                except Exception:
                    qlist = []
            negative_questions.extend(random.sample(qlist, min(num_questions, len(qlist))))

        # If more than num_questions, randomly select num_questions
        if len(negative_questions) > num_questions:
            negative_questions = random.sample(negative_questions, num_questions)
        
        return negative_questions

    def get_negative_concepts(self, concept_class=None, exclude_concepts=None):
        """
        Returns a list of available concept classes for negative sampling, excluding the provided concept_class and any classes in exclude_concepts, as well as any classes with no questions.
        :param concept_class: The concept to exclude from the list (single string).
        :param exclude_concepts: List of concepts to exclude from the list.
        :return: List of concept class names.
        """
        if not hasattr(self, 'reasoning_questions'):
            raise ValueError("Reasoning questions have not been loaded. Call load(path_to_csv) first.")
        if exclude_concepts is None:
            exclude_concepts = []
        exclude_set = set(exclude_concepts)
        if concept_class is not None:
            exclude_set.add(concept_class)
        available_classes = [
            c for c in self.reasoning_questions.keys()
            if c not in exclude_set and self._has_questions(self.reasoning_questions[c]["vlm_reasoning_questions"])
        ]
        return available_classes

    def _has_questions(self, questions):
        # Helper to check if a class has any questions (non-empty list after parsing)
        if isinstance(questions, dict) and 'questions' in questions:
            qlist = questions['questions']
        elif isinstance(questions, str):
            try:
                qlist = json.loads(questions)
            except Exception:
                return False
        else:
            qlist = questions
        if isinstance(qlist, str):
            try:
                qlist = json.loads(qlist)
            except Exception:
                return False
        return bool(qlist) and isinstance(qlist, list) and len(qlist) > 0
    

    def get_embedding(self, text):
        # ensure model is loaded (and device is set)
        if self.model is None or self.tokenizer is None:
            self._load_language_model()

        inputs = self.tokenizer(text, return_tensors="pt")
        # move inputs to saved device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        # Ensure embedding is on the same device as model
        embedding = embedding.to(self.device)
        return embedding

    def find_top_k_similar(self, target, concepts, k=5):
        target_emb = self.get_embedding(target).unsqueeze(0)
        all_embeddings = torch.stack([self.get_embedding(concept) for concept in concepts])
        # Move tensors to CPU before converting to numpy
        similarities = cosine_similarity(target_emb.cpu().numpy(), all_embeddings.cpu().numpy())[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(concepts[i], similarities[i]) for i in top_k_indices]

    def find_top_k_dissimilar(self, target, concepts, k=5):
        target_emb = self.get_embedding(target).unsqueeze(0)
        all_embeddings = torch.stack([self.get_embedding(concept) for concept in concepts])
        # Move tensors to CPU before converting to numpy
        similarities = cosine_similarity(target_emb.cpu().numpy(), all_embeddings.cpu().numpy())[0]
        top_k_indices = np.argsort(similarities)[:k]
        return [(concepts[i], similarities[i]) for i in top_k_indices]
