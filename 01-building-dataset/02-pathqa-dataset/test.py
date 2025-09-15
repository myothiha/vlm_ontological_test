import sys
import os

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.OntologyBenchmarkBuilder.pipeline import OntoBenchPipeline

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