import os
import json
import pandas as pd
from libs.prompt.prompt_manager import PromptTemplateManager

class KnowledgeExtractor:
    """
    A class to extract ontological knowledge for each concept using an LLM and prompt templates.
    """
    def __init__(self, llm, result_extract_func, prompt_dir="prompt_templates",  prompt_template="02_generate_knowledge_prompt1", output_dir="results"):
        """
        :param llm: The language model to use for knowledge extraction.
        :param prompt_dir: Directory containing prompt templates.
        :param output_dir: Directory to save results.
        :param prompt_template: The prompt template to use for knowledge extraction.
        """
        self.llm = llm
        self.result_extract_func = result_extract_func
        self.prompt_template_dir = prompt_templates
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename = os.path.join(self.output_dir, "02_ontological_knowledge_one_shot.csv")
        self.prompt_template = prompt_template

    def _setup_templates(self):
        """
        Loads the prompt templates from the specified directory.
        """
        print("Setting up prompt templates...", self.prompt_template_dir)
        prompt_dir = os.path.join("", self.prompt_template_dir)
        print("Prompt Directory:", prompt_dir)
        self.manager = PromptTemplateManager(prompt_dir=prompt_dir)

        self.prompt_templates = []
        
        # load templates files.txt files from the prompt directory.
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".txt"):
                template_name = filename[:-4]
                self.prompt_templates.append(template_name)

        print(f"Loaded {len(self.prompt_templates)} prompt templates from {self.prompt_template_dir}")

    def extract_knowledge(self, queries_csv):
        """
        Extracts and saves ontological knowledge for each concept/class in the queries CSV.
        """
        # Load previously saved questions
        df = pd.read_csv(queries_csv)

        # Write header once
        generated_objects = []
        if os.path.exists(self.output_filename):
            df_onto_queries = pd.read_csv(self.output_filename)
            if 'class' in df_onto_queries:
                generated_objects = df_onto_queries['class'].to_list()
        else:
            pd.DataFrame(columns=["class", "knowledge_questions", "generated_knowledge"]).to_csv(self.output_filename, index=False)

        for _, row in df.iterrows():
            class_name = row["class"]
            knowledge_questions = json.loads(row["knowledge_questions"])
            if class_name in generated_objects:
                continue

            # Iterate the five dimension of conceptual KQs.
            generated_knowledge = dict()
            for knowledge_type, questions in knowledge_questions.items():
                print(f"🔄 Generating {knowledge_type} Knowledge for: {class_name}")
                prompt = self.manager.format(self.prompt_template, class_name=class_name, questions_json=json.dumps(questions, indent=2))
                response = self.llm(prompt, max_new_tokens=2048)
                print("LLM Response", response)
                generated_knowledge[knowledge_type] = self.result_extract_func(response)
                
            generated_objects.append(class_name)

            print("Generated Knowledge", generated_knowledge)
            row_df = pd.DataFrame([{
                "class": class_name,
                "knowledge_questions": json.dumps(knowledge_questions),
                "generated_knowledge": json.dumps(generated_knowledge),
            }])
            row_df.to_csv(self.output_filename, mode='a', header=False, index=False)
            print(f"✅ Saved: {class_name}")

        return self.output_filename

    def extract_knowledge_for_single_concept(self, concept, knowledge_questions):
        # Iterate the five dimension of conceptual KQs.
        generated_knowledge = dict()
        for knowledge_type, questions in knowledge_questions.items():
            print(f"🔄 Generating {knowledge_type} Knowledge for: {concept}")
            prompt = self.manager.format(self.prompt_template, class_name=concept, questions_json=json.dumps(questions, indent=2))
            response = self.llm(prompt, max_new_tokens=2048)
            print("LLM Response", response)
            generated_knowledge[knowledge_type] = self.result_extract_func(response)
            
        return generated_knowledge