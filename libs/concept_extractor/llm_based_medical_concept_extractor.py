from libs.concept_extractor.abstract_concept_extractor import AbstractConceptExtractor
from pathlib import Path
import os
import json
import re
from libs.prompt.prompt_manager import PromptTemplateManager

class LLMBasedMedicalConceptExtractor(AbstractConceptExtractor):
    def __init__(self, model, backup_extractor=None):
        self.llm = model
        self.backup_extractor: AbstractConceptExtractor = backup_extractor

        self.current_dir = Path(__file__).resolve().parent
        prompt_dir = os.path.join(self.current_dir, "prompt_template")

        self.prompt_manager = PromptTemplateManager(prompt_dir=prompt_dir)

    def extract(self, text) -> list:
        """
        Extracts medical concepts from the given text using the LLM model.
        
        :param text: The input text from which to extract medical concepts.
        :return: A list of extracted medical concepts.
        """
        # This method should implement the logic to use the LLM model to extract concepts
        # For now, we return an empty list as a placeholder

        self.text = text

        prompt = self.prompt_manager.format("extract_medical_concepts", text=text)

        response = self.llm(prompt, max_new_tokens=500)

        # print("####################Extraction Start####################")
        # print("LLM Response:")
        # print(response)
        # print("####################Extraction End####################")

        return self.extract_from_llm(response)
    
    def backup_extract(self):
        result = self.backup_extractor.extract(self.text)
        return result

    def extract_from_llm(self, llm_response: str):
        
        """
        Extract medical concepts list from LLM response, fixing malformed lists if necessary.
        
        Args:
            llm_response (str): Full LLM response including the result block.
            
        Returns:
            list: List of medical concept strings.
        """
        try:
            # Step 1: Remove everything before 'Output:'
            output_text = llm_response.split("### End of Example", 1)[-1].strip()

            # Step 2: Extract <result>[...]</result> content
            match = re.search(r"<result>\s*(\[[\s\S]*?\])\s*</result>", output_text)
            if not match:
                print("No valid <result> found in the response.")
                print("llm_response:", llm_response)
                return []
            
            raw_list = match.group(1)

            # Step 3: Try JSON parsing first
            try:
                result_list = json.loads(raw_list)
                if len(result_list) == 0:
                    print("Empty list detected in the response.")

                return result_list
            except json.JSONDecodeError:
                # Step 4: Fallback: Manually convert to JSON array
                # Remove brackets and split by comma
                print("Malformed JSON detected, attempting to fix it.")
                inner = raw_list.strip()[1:-1]
                items = [item.strip() for item in inner.split(',') if item.strip()]
                quoted_items = [f'"{item}"' for item in items]
                fixed_json = '[' + ', '.join(quoted_items) + ']'
                return json.loads(fixed_json)
        except Exception as e:
            print(f"Error while extracting concepts: {e}")
            if self.backup_extractor:
                print(f"Falling back to the backup extractor.")
                return self.backup_extract()
            return []


    def extract_from_llm_bk(self, llm_response: str) -> list:
        """
        Extracts medical concepts from the LLM response.
        
        :param response: The response from the LLM containing medical concepts.
        :return: A list of extracted medical concepts.
        """
        
        try:
            # Step 1: Remove everything before 'Output:'
            output_text = llm_response.split("### End of Example", 1)[-1].strip()

            # Step 2: Extract content inside <result>...</result>
            match = re.search(r"<result>\s*(\[[\s\S]*?\])\s*</result>", output_text)
            if not match:
                return []
            
            # Step 3: Parse the JSON list
            concept_list_str = match.group(1)
            concepts = json.loads(concept_list_str)
            
            # Ensure it's a list of strings
            if isinstance(concepts, list) and all(isinstance(item, str) for item in concepts):
                return concepts
            else:
                print(concept_list_str)
                return []
        except Exception as e:
            print(llm_response)
            print(f"Error while extracting concepts: {e}. Switching to backup extractor.")
            return llm_response