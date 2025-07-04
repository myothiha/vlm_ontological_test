# llm_loader/llm_wrapper.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration

class LVLMWrapper:
    def __init__(self, model_path, pipeline_kwargs=None):
        self.model_path = model_path
        self.processor = None
        self.model = None
        # self._build_pipeline(pipeline_kwargs or {})

    def _load(self):

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _build_pipeline(self, pipeline_kwargs):
        default_kwargs = {
            "task": "text-generation",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "truncation": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        default_kwargs.update(pipeline_kwargs)
        self.pipe = pipeline(**default_kwargs)

    def __call__(self, image: str, prompt: str, **generate_kwargs):
        if self.model is None or self.processor is None:
            self._load()

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_pipeline(self):
        return self.pipe