from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.OntologyBenchmarkBuilder.knowledge_question_generator import KQGenerator
from libs.utils import extract_list_from_gpt, extract_list_from_ollama_models, extract_list_from_hf_models

# llm = GPTLLMWrapper("gpt-4.1")

# Example usage
# ollama = OllamaWrapper(model="llava:34b", multi_turn=True)
# ollama.set_system_prompt("You are a helpful assistant.") # only for multi-turn mode

# print("Single-turn generation:")
# response = ollama("What is the capital of France?", temperature=0.1)
# print(response)

# print("\nMulti-turn chat:")
# response = ollama("Hello, who won the World Cup in 2018?", temperature=0.1)
# print(response)
# response = ollama("Where was it held?", temperature=0.1)
# print(response)

from PIL import Image
img = Image.open("example.jpg")

ollama = OllamaWrapper(model="llava:34b")
print("Image Type:", type(img))
multimodal_response = ollama(
    images=[img],
    prompt="Describe the main object and its condition.",
    # temperature=0.1
)
print(multimodal_response)