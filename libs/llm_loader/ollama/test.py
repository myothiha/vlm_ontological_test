from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

# conceptExtractor = LLMBasedMedicalConceptExtractor(model="Bio_GPT_Large")
# conceptExtractor.extract("Hello, I have a headache and fever. I also have a sore throat and runny nose. My doctor prescribed ibuprofen and recommended rest. I will get a blood test tomorrow to check for any infections.")

model_name = "taozhiyuai/openbiollm-llama-3:70b_q2_k"
print("Using model:", model_name)

stream = False
multi_turn = True
llm = OllamaWrapper(model=model_name, multi_turn=multi_turn)

query = "What is the role of the liver in the human body?"
print("Query:", query)

if multi_turn:
    # Multi Turn Mode
    llm.set_system_prompt("You are a helpful medical assistant.")
    questions = [
        "What is the role of the liver in the human body?",
        "What are the symptoms of liver disease?",
        "How does the liver process toxins?",
    ]
    for question in questions:
        print("Question:", question)
        result = llm(question, stream=stream)

        if stream:
            for token in result:
                print(token, end='', flush=True)
        else:
            print("Result:", result)

        print("\nDone.")

        print("History:")
        print(llm.history)
else:
    # Single Turn Mode
    result = llm(query, stream=stream)

    if stream:
        for token in result:
            print(token, end='', flush=True)
    else:
        print("Result:", result)

    print("\nDone.")