import os
import json
import pandas as pd
from libs.prompt.prompt_manager import PromptTemplateManager
from libs.utils import extract_list

class ReasoningQuestionGenerator:
    """
    A class to generate reasoning questions based on a given ontology and previously generated knowledge.
    """
    def __init__(self, llm, result_extract_func, prompt_dir="prompt_templates", output_dir="results", prompt_template="03_vlm_reasoning_questions_one_shot"):
        """
        :param llm: The language model to use for reasoning question generation.
        :param prompt_dir: Directory containing prompt templates.
        :param output_dir: Directory to save results.
        :param prompt_template: The prompt template to use for reasoning question generation.
        """
        self.llm = llm
        self.result_extract_func = result_extract_func
        self.manager = PromptTemplateManager(prompt_dir=prompt_dir)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.prompt_template = prompt_template
        self.output_filename = os.path.join(self.output_dir, "03_generate_vlm_reasoning_questions.csv")

    def generate_reasoning_questions(self, knowledge_csv="02_ontological_knowledge_one_shot.csv"):
        """
        Generates and saves reasoning questions for each class/concept using the LLM and prompt template.
        """
        # Load generated ontological knowledge
        df = pd.read_csv(knowledge_csv)

        # Write header once
        generated_objects = []
        if os.path.exists(self.output_filename):
            df_reasoning_questions = pd.read_csv(self.output_filename)
            if 'class' in df_reasoning_questions:
                generated_objects = df_reasoning_questions['class'].to_list()
        else:
            pd.DataFrame(columns=["class", "knowledge_questions", "generated_knowledge", "vlm_reasoning_questions"]).to_csv(self.output_filename, index=False)

        for _, row in df.iterrows():
            class_name = row["class"]
            knowledge_questions = json.loads(row["knowledge_questions"])
            generated_knowledge = json.loads(row["generated_knowledge"])
            if class_name in generated_objects:
                continue
            generated_objects.append(class_name)
            print(f"🔄 Generating Reasoning Questions for: {class_name}")
            prompt = self.manager.format(self.prompt_template, class_name=class_name, generated_knowledge=json.dumps(generated_knowledge, indent=2))
            response = self.llm(prompt, max_new_tokens=600)
            print("LLM Response", response)
            vlm_reasoning_questions = self.result_extract_func(response)
            print("Generated VLM Reasoning Questions", vlm_reasoning_questions)
            row_df = pd.DataFrame([{
                "class": class_name,
                "knowledge_questions": json.dumps(knowledge_questions),
                "generated_knowledge": json.dumps(generated_knowledge),
                "vlm_reasoning_questions": json.dumps(vlm_reasoning_questions),
            }])
            row_df.to_csv(self.output_filename, mode='a', header=False, index=False)
            print(f"✅ Saved: {class_name}")

        return self.output_filename
