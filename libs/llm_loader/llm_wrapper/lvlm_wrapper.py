# llm_loader/llm_wrapper.py

from __future__ import annotations
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    pipeline,
    AutoModelForImageTextToText,
)

class LVLMWrapper:
    def __init__(self, model_path, *, quantisation: int | None = None, pipeline_kwargs:dict | None = None):
        if quantisation not in (None, 8, 4):
            raise ValueError("quantisation must be None, 8, or 4")
        
        self.model_path = model_path
        self.quantisation = quantisation
        self.pipeline_kwargs = pipeline_kwargs or {}

        self.processor = None
        self.model = None
        self.tokenizer = None  # (Gemma3 uses the processor’s tokenizer internally)

    def _load(self):

        # ----------------------------------------
        # 1. Build the correct kwargs
        # ----------------------------------------
        kwargs: dict = {"device_map": "auto"}

        if self.quantisation == 8:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )

        elif self.quantisation == 4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_path)

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
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False,
                **generate_kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_pipeline(self):
        return self.pipe