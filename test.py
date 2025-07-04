from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor


conceptExtractor = LLMBasedMedicalConceptExtractor(model="Bio_GPT_Large")
conceptExtractor.extract("Hello, I have a headache and fever. I also have a sore throat and runny nose. My doctor prescribed ibuprofen and recommended rest. I will get a blood test tomorrow to check for any infections.")