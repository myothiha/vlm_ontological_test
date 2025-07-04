import re
import json

from libs.prompt.prompt_manager import PromptTemplateManager
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper

class QAEvaluationLLM:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.promptManager = PromptTemplateManager(prompt_dir="prompt_templates")
        # Load model
        self.llm = LLMWrapper(model_path=model_path)

    def evaluate(self, question, ref_answer, cand_answer):
        prompt = self.promptManager.format("pathvqa_answer_evaluation", question=question, ref_answer=ref_answer, model_answer=cand_answer)
        response = self.llm(prompt, max_new_tokens=128)
        print("LLM Response:", response)

        result = self.extract_first_valid_output_json(response)
        result["score"] = self.normalize_score(result)
        return result
    
    # normalize scores from 0 to 1
    def normalize_score(self, result):
        """
        The value 1, 2, 3 to the range 0.0 to 1.0.
        """
        if result["rating"] == 1:
            return 0.0
        elif result["rating"] == 2:
            return 0.5
        elif result["rating"] == 3:
            return 1.0
        else:
            raise ValueError(f"Unexpected rating value: {result['rating']}")

    def extract_first_valid_output_json(self, text: str):
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