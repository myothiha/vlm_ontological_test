import sys
import os
import json

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.OntologyBenchmarkBuilder.knowledge_question_generator import KQGenerator
from libs.OntologyBenchmarkBuilder.knowledge_extractor import KnowledgeExtractor
from libs.OntologyBenchmarkBuilder.reasoning_question_generator import ReasoningQuestionGenerator
from libs.utils import extract_list_from_gpt, extract_list_from_ollama_models, extract_list_from_hf_models

class OntoBenchPipeline:
    def __init__(
            self, 
            llm, 
            knowledge_question_prompt_templates, 
            generate_knowledge_prompt_template, 
            vlm_reasoning_prompt_template
        ):
        # self.llm = llm
        # self.knowledge_question_prompt_templates = knowledge_question_prompt_templates
        # self.generate_knowledge_prompt_template = generate_knowledge_prompt_template
        # self.vlm_reasoning_prompt_template = vlm_reasoning_prompt_template

        self.kq_generator = KQGenerator(
            llm=llm,
            result_extract_func=extract_list_from_gpt,
            prompt_templates=knowledge_question_prompt_templates,
        )

        self.knowledge_extractor = KnowledgeExtractor(
            llm=llm,
            result_extract_func=extract_list_from_gpt,
            prompt_templates=generate_knowledge_prompt_template,
        )

        self.rq_generator = ReasoningQuestionGenerator(
            llm=llm,
            result_extract_func=extract_list_from_gpt,
            prompt_templates=vlm_reasoning_prompt_template,
        )

    def generate_knowledge(self, concept, knowledge_questions):
        knowledge = self.knowledge_extractor.extract_knowledge_for_single_concept(concept, knowledge_questions)
        return knowledge

    def __call__(self, concept):
        knowledge_questions = self.kq_generator.generate_questions_for_single_concept(concept)
        knowledge = self.knowledge_extractor.extract_knowledge_for_single_concept(concept, knowledge_questions)
        vlm_reasoning_questions = self.rq_generator.generate_reasoning_questions_for_single_concept(concept, knowledge)

        return {
            "class": concept,
            "knowledge_questions": json.dumps(knowledge_questions),
            "generated_knowledge": json.dumps(knowledge),
            "vlm_reasoning_questions": json.dumps(vlm_reasoning_questions)
        }

if __name__ == "__main__":

    # llm = GPTLLMWrapper("gpt-4.1")
    llm = OllamaWrapper(model="gpt-oss:20b")

    pipeline = OntoBenchPipeline(
        llm = llm,
        knowledge_question_prompt_templates = "prompt_templates/kq_prompt_templates",
        generate_knowledge_prompt_template = "02_generate_knowledge_prompt1",
        vlm_reasoning_prompt_template = "03_vlm_reasoning_questions_one_shot"
    )

    # Example code
    concept = "Liver Stem Cell"
    print(pipeline(concept))