import re
import json

from libs.prompt.prompt_manager import PromptTemplateManager

class QAEvaluationLLM:
    
    def __init__(self, llm):
        self.promptManager = PromptTemplateManager(prompt_dir="prompt_templates")
        # Load model
        self.llm = llm

    def evaluate(self, question, ref_answer, cand_answer):
        prompt = self.promptManager.format("pathvqa_answer_evaluation", question=question, ref_answer=ref_answer, model_answer=cand_answer)
        response = self.llm(prompt, max_new_tokens=128)

        result = self.extract_first_valid_output_json2(response)
        if result is None:
            return {
                "score": 0, 
                "rating": 0, 
                "comment": "No Valid Output Json is found", 
                "raw_response": response
            }
        
        result["score"] = self.normalize_score(result)
        return result
    
    # normalize scores from 0 to 1
    def normalize_score(self, result):
        """
        The value 1, 2, 3 to the range 0.0 to 1.0.
        """
        if result is None or "rating" not in result:
            print("No rating found in result.")
            return None
        else:
            if result["rating"] == 1:
                return 0.0
            elif result["rating"] == 2:
                return 0.5
            elif result["rating"] == 3:
                return 1.0
            else:
                print(f"Unexpected rating value: {result['rating']}")
                return 0

    def extract_first_valid_output_json1(self, text: str):
        """
        Scan all <result>…</result> blocks in `text`.  
        Starting from the second block, extract the JSON after 'Output:' and attempt to parse it.
        Return the first successfully loaded JSON object, or print a message if none are valid.
        """
        # 1) grab all result‐blocks
        blocks = re.findall(r"<result>(.*?)</result>", text, flags=re.DOTALL)
        # 2) iterate starting at the second one
        for blk in blocks[1:]:
            # find the {...} JSON after 'Output:'
            m = re.search(r"Output:\s*({[\s\S]*?})", blk)

            # if no match, skip to next block
            if not m:
                continue
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # invalid JSON → skip to next block
                continue

        # if we get here, nothing parsed correctly
        print("No Valid Output Json is found")
        return None
    
    def extract_first_valid_output_json2(self, text: str):
        # extract json object from the string.
        json_str = re.search(r"\s*({[\s\S]*?})", text)
        print("json_str:", json_str)    
        if json_str:
            try:
                return json.loads(json_str.group(0))
            except json.JSONDecodeError:
                print("Invalid JSON format." + text)
                return None
        return None