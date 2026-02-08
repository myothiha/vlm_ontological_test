import os
import pandas as pd
import sqlite3
import json
import re

from libs.concept_extractor.abstract_concept_extractor import AbstractConceptExtractor
from libs.dataset_loader.dataset import Dataset
from libs.OntologyBenchmarkBuilder.knowledge_question_generator import KQGenerator
from libs.OntologyBenchmarkBuilder.knowledge_extractor import KnowledgeExtractor
from libs.OntologyBenchmarkBuilder.reasoning_question_generator import ReasoningQuestionGenerator
from libs.concept_extractor.medical_concept_classifier import MedicalConceptClassifier
from libs.utils import clean_special_chars, setup_logger

class BenchmarkBuilder:
    def __init__(self,dataset: Dataset, llm, result_extract_func, knowledge_question_prompt_templates, generate_knowledge_prompt_template, vlm_reasoning_prompt_template, concept_extractor: AbstractConceptExtractor, medical_concept_classifier: MedicalConceptClassifier, required_concept_extraction = False, output_dir: str = "results"):
        self.dataset = dataset
        self.llm = llm
        self.result_extract_func = result_extract_func
        self.concept_extractor = concept_extractor
        self.medical_concept_classifier = medical_concept_classifier
        self.required_concept_extraction = required_concept_extraction
        self.output_dir = output_dir
        self.prompt1 = knowledge_question_prompt_templates
        self.prompt2 = generate_knowledge_prompt_template
        self.prompt3 = vlm_reasoning_prompt_template
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger("BenchmarkBuilder", os.path.join(self.output_dir, "benchmark_builder.log"))
    
    def setConceptExtractor(self, concept_extractor: AbstractConceptExtractor):
        """
        Set the concept extractor to be used for this benchmark.
        
        :param concept_extractor: An instance of AbstractConceptExtractor.
        """
        self.concept_extractor = concept_extractor
        self.required_concept_extraction = True

    def build(self):
        # Placeholder for building the benchmark
        self.logger.info(f"Building benchmark....")
        
        self.logger.info("################################################################################")
        self.logger.info("########################## Step 0: Concept Extractions #########################")
        self.logger.info("################################################################################")
        
        # Step 0: Extract concepts if required
        concepts = self.get_concepts()
        self.logger.info(f"Extracted {len(concepts)} unique concepts.")
        # print(f"Concepts: {concepts}")

        self.logger.info("##########################################################################################")
        self.logger.info("########################## Step 1: Knowledge Question Generation #########################")
        self.logger.info("##########################################################################################")
        
        # Step 1: Generate knowledge questions
        kq_generator = KQGenerator(
            llm=self.llm,
            result_extract_func=self.result_extract_func,
            prompt_dir=self.prompt1,
            prompt_templates=self.prompt1,
            output_dir=self.output_dir
        )
        query_csv = kq_generator.generate_questions(concepts=concepts)
        
        self.logger.info("##########################################################################################")
        self.logger.info("########################## Step 2: Knowledge Extraction from LLM #########################")
        self.logger.info("##########################################################################################")

        # Step 2: Extract ontological knowledge for each concept
        knowledge_extractor = KnowledgeExtractor(
            llm=self.llm,
            result_extract_func=self.result_extract_func,
            prompt_dir=self.prompt2,
            prompt_template=self.prompt2,
            output_dir=self.output_dir,
        )
        extracted_knowledge_csv = knowledge_extractor.extract_knowledge(queries_csv=query_csv)
        
        
        reasoning_question_csv = []
        # print("##############################################################################################")
        # print("########################## Step 3: VLM Reasoning Question Generation #########################")
        # print("##############################################################################################")
        # # Step 3: Generate reasoning questions based on the extracted knowledge
        # reasoning_question_generator = ReasoningQuestionGenerator(
        #     llm=self.llm,
        #     result_extract_func=self.result_extract_func,
        #     prompt_dir="prompt_templates",
        #     prompt_template=self.prompt3,
        #     output_dir=self.output_dir,
        # )
        # reasoning_question_csv = reasoning_question_generator.generate_reasoning_questions(extracted_knowledge_csv)
        
        return reasoning_question_csv
        

    def get_concepts(self) -> list:
        """
        Extract concepts from the dataset using the concept extractor.
        
        :return: A list of extracted concepts.
        """
        output_file = os.path.abspath(os.path.join(self.output_dir, "00_unique_concepts.txt"))

        # create text, concepts pair for human annotation for concept extraction
        output_log_csv = os.path.abspath(os.path.join(self.output_dir, "00_concept_extraction.csv"))
        # output_log_db  = os.path.abspath(os.path.join(self.output_dir, "result.db"))

        if self.required_concept_extraction:
            # Check if the output file already exists

            # if os.path.exists(output_file):
            #     # Load medical concept list from results/unique_concepts.txt
            #     with open(output_file, "r") as f:
            #         concepts = [line.strip() for line in f.readlines() if line.strip()]
            #     return concepts

            # Check if the output log CSV file exists
            processed_texts = []
            if os.path.exists(output_log_csv):
                current_df = pd.read_csv(output_log_csv)
                processed_texts = current_df['text'].to_list()
            else:
                pd.DataFrame(columns=["text", "concepts"]).to_csv(output_log_csv, index=False)

            # If concept extractor is not set, raise an error
            if not self.concept_extractor:
                raise ValueError("Concept extractor is not set.")
            
            # Extract unique concepts from the dataset
            self.logger.info("Extracting concepts...")
            unique_concepts = set()
            for item in self.dataset:
                text = item.get("text", "")

                # If the text has been already processed, use the cached concepts.
                if text in processed_texts:
                    raw_concepts = current_df[current_df['text'] == text]['concepts'].to_list()
                    # print("Raw concepts:", raw_concepts[0])
                    # concepts = json.loads(raw_concepts) if raw_concepts else []
                    concepts = eval(raw_concepts[0]) if raw_concepts[0].startswith("[") else [raw_concepts[0]]
                else:
                    concepts = self.concept_extractor.extract(text)

                    for concept in concepts:
                        is_medical = self.medical_concept_classifier.classify(concept)
                        if not is_medical:
                            self.logger.info(f"{concept} is not a medical-concept. Removed.")
                            concepts.remove(concept)
                        # print(f"{concept}:", is_medical)

                    log_row = pd.DataFrame([
                        {"text": text,
                        "concepts": concepts}
                    ])
                    log_row.to_csv(output_log_csv, mode='a', header=False, index=False)

                unique_concepts.update(concepts)

                # print("Input text:", text)
                self.logger.info(f"Extracted {len(concepts)}: {concepts}")

            # Clean unique concepts: lower case, remove double space, remove special characters.
            unique_concepts = {clean_special_chars(concept) for concept in unique_concepts}
            unique_concepts = {concept.strip() for concept in unique_concepts}

            unique_concepts = list(sorted(unique_concepts))
            # Save unique concepts to disk
            # Save unique concepts to disk
            with open(output_file, "w") as f:
                self.logger.info(output_file)
                for concept in unique_concepts:
                    f.write(concept + "\n")

            self.logger.info(f"Extracted {len(unique_concepts)} unique concepts.")

            # table = "00_concept_extraction"
            # con = sqlite3.connect(output_log_db)
            # # For 20k rows you can load at once; for bigger, add chunksize (see below)
            # df = pd.read_csv(output_log_csv)            # add dtype=... if you want control
            # df.to_sql(table, con, if_exists="replace", index=False)
            # con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_id ON {table}(id);")  # optional
            # con.close()

            return unique_concepts
        else:
            all_concepts = self.dataset.get_all_texts()
            unique_concepts = set()
            for concepts in all_concepts:
                unique_concepts.update(concepts)

            # Save unique concepts to disk
            with open(output_file, "w") as f:
                self.logger.info(output_file)
                for concept in sorted(unique_concepts):
                    f.write(concept + "\n")

            return list(sorted(unique_concepts))