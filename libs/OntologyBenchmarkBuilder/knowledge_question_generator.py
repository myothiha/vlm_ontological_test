import os
import json
import pandas as pd
from libs.prompt.prompt_manager import PromptTemplateManager

class KQGenerator:
    """
    A class to generate knowledge questions based on a given ontology.
    """
    def __init__(self, llm, result_extract_func, prompt_dir="prompt_templates", prompt_templates="01_few_shot_without_instruction", output_dir="results"):
        """
        Initializes the KQGenerator with a set of concepts, an LLM, and prompt manager.
        :param concepts: The concepts to generate questions from.
        :param llm: The language model to use for question generation.
        :param prompt_dir: Directory containing prompt templates.
        :param output_dir: Directory to save results.
        """
        self.llm = llm
        self.result_extract_func = result_extract_func
        self.prompt_template_dir = prompt_templates
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_filename = os.path.join(self.output_dir, "01_ontological_queries.csv")
        
        self._setup_templates()

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

    def generate_questions(self, concepts):
        """
        Generates and saves knowledge questions for each concept.
        """
        # Load already generated objects if file exists
        
        generated_objects = []
        if os.path.exists(self.csv_filename):
            df_onto_queries = pd.read_csv(self.csv_filename)
            generated_objects = df_onto_queries['class'].to_list()
        else:
            pd.DataFrame(columns=["class", "knowledge_questions"]).to_csv(self.csv_filename, index=False)

        for class_name in concepts:
    
            if class_name in generated_objects:
                continue
            
            knowledge_questions = dict()
            for prompt_template in self.prompt_templates:    
                print(f"🔄 Generating {prompt_template} Knowledge Questions for: {class_name}")

                prompt = self.manager.format(prompt_template, class_name=class_name)
                response = self.llm(prompt, max_new_tokens=2048)
                print("Response", response)
                knowledge_questions[prompt_template] = self.result_extract_func(response)

            if "Error" in knowledge_questions:
                print(f"❌ Error extracting knowledge questions for {class_name}: {knowledge_questions}")
                continue
            
            if len(knowledge_questions) == 0:
                print(f"❌ No knowledge questions generated for {class_name}.")
                continue
            
            print("======Done======")
            print("Knowledge Questions:", knowledge_questions)
            
            result = {
                "class": class_name,
                "knowledge_questions": json.dumps(knowledge_questions)
            }

            # Append to CSV file
            row_df = pd.DataFrame([result])
            row_df.to_csv(self.csv_filename, mode='a', header=False, index=False)

            print(f"✅ Saved: {class_name}")

        return self.csv_filename
    
    def generate_questions_for_single_concept(self, concept):
        knowledge_questions = dict()
        for prompt_template in self.prompt_templates:    
            print(f"🔄 Generating {prompt_template} Knowledge Questions for: {concept}")

            prompt = self.manager.format(prompt_template, class_name=concept)
            response = self.llm(prompt, max_new_tokens=2048)
            print("Response", response)
            knowledge_questions[prompt_template] = self.result_extract_func(response)

        if "Error" in knowledge_questions:
            print(f"❌ Error extracting knowledge questions for {concept}: {knowledge_questions}")
        
        if len(knowledge_questions) == 0:
            print(f"❌ No knowledge questions generated for {concept}.")
        
        print("======Done======")
        print("Knowledge Questions:", knowledge_questions)
        
        return knowledge_questions