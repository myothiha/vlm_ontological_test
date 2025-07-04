from libs.concept_extractor.abstract_concept_extractor import AbstractConceptExtractor
from libs.dataset_loader.dataset import Dataset
from libs.OntologyBenchmarkBuilder.knowledge_question_generator import KQGenerator
from libs.OntologyBenchmarkBuilder.knowledge_extractor import KnowledgeExtractor
from libs.OntologyBenchmarkBuilder.reasoning_question_generator import ReasoningQuestionGenerator
import os
class BenchmarkBuilder:
    def __init__(self, dataset: Dataset, llm, result_extract_func, knowledge_question_prompt_template, generate_knowledge_prompt_template, vlm_reasoning_prompt_template, concept_extractor: AbstractConceptExtractor = None, required_concept_extraction = False, output_dir: str = "results"):
        self.dataset = dataset
        self.llm = llm
        self.result_extract_func = result_extract_func
        self.concept_extractor = concept_extractor
        self.required_concept_extraction = required_concept_extraction
        self.output_dir = output_dir
        self.prompt1 = knowledge_question_prompt_template
        self.prompt2 = generate_knowledge_prompt_template
        self.prompt3 = vlm_reasoning_prompt_template
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setConceptExtractor(self, concept_extractor: AbstractConceptExtractor):
        """
        Set the concept extractor to be used for this benchmark.
        
        :param concept_extractor: An instance of AbstractConceptExtractor.
        """
        self.concept_extractor = concept_extractor
        self.required_concept_extraction = True

    def build(self):
        # Placeholder for building the benchmark
        print(f"Building benchmark....")


        print("################################################################################")
        print("########################## Step 0: Concept Extractions #########################")
        print("################################################################################")
        
        # Step 0: Extract concepts if required
        concepts = self.get_concepts()
        print(f"Extracted {len(concepts)} unique concepts.")
        print(f"Concepts: {concepts}")

        print("##########################################################################################")
        print("########################## Step 1: Knowledge Question Generation #########################")
        print("##########################################################################################")

        # Step 1: Generate knowledge questions
        kq_generator = KQGenerator(
            llm=self.llm,
            result_extract_func=self.result_extract_func,
            prompt_dir="prompt_templates",
            prompt_template=self.prompt1,
            output_dir=self.output_dir
        )
        query_csv = kq_generator.generate_questions(concepts=concepts)
        
        print("##########################################################################################")
        print("########################## Step 2: Knowledge Extraction from LLM #########################")
        print("##########################################################################################")

        # Step 2: Extract ontological knowledge for each concept
        knowledge_extractor = KnowledgeExtractor(
            llm=self.llm,
            result_extract_func=self.result_extract_func,
            prompt_dir="prompt_templates",
            prompt_template=self.prompt2,
            output_dir=self.output_dir,
        )
        extracted_knowledge_csv = knowledge_extractor.extract_knowledge(queries_csv=query_csv)
        
        print("##############################################################################################")
        print("########################## Step 3: VLM Reasoning Question Generation #########################")
        print("##############################################################################################")
        # Step 3: Generate reasoning questions based on the extracted knowledge
        reasoning_question_generator = ReasoningQuestionGenerator(
            llm=self.llm,
            result_extract_func=self.result_extract_func,
            prompt_dir="prompt_templates",
            prompt_template=self.prompt3,
            output_dir=self.output_dir,
        )
        reasoning_question_csv = reasoning_question_generator.generate_reasoning_questions(extracted_knowledge_csv)
        
        return reasoning_question_csv

    def get_concepts(self) -> list:
        """
        Extract concepts from the dataset using the concept extractor.
        
        :return: A list of extracted concepts.
        """
        output_file = os.path.abspath(os.path.join(self.output_dir, "00_unique_concepts.txt"))

        if self.required_concept_extraction:
            # Check if the output file already exists

            if os.path.exists(output_file):
                # Load medical concept list from results/unique_concepts.txt
                with open(output_file, "r") as f:
                    concepts = [line.strip() for line in f.readlines() if line.strip()]
                return concepts

            # If concept extractor is not set, raise an error
            if not self.concept_extractor:
                raise ValueError("Concept extractor is not set.")
            
            # Extract unique concepts from the dataset
            print("Extracting concepts...")
            unique_concepts = set()
            for item in self.dataset:
                text = item.get("text", "")
                concepts = self.concept_extractor.extract(text)
                # Add only the .text of each entity to the set
                unique_concepts.update(ent.text for ent in concepts)

            # Save unique concepts to disk
            with open(output_file, "w") as f:
                print(output_file)
                for concept in sorted(unique_concepts):
                    f.write(concept + "\n")

            print(f"Extracted {len(unique_concepts)} unique concepts.")
            return list(sorted(unique_concepts))
        else:
            all_concepts = self.dataset.get_all_texts()
            unique_concepts = set()
            for concepts in all_concepts:
                unique_concepts.update(concepts)

            # Save unique concepts to disk
            with open(output_file, "w") as f:
                print(output_file)
                for concept in sorted(unique_concepts):
                    f.write(concept + "\n")

            return list(sorted(unique_concepts))