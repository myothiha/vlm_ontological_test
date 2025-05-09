# llm_loader/llm_wrapper.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

class LLMWrapper:
    def __init__(self, model_path, pipeline_kwargs=None, quantized=False, quantization_bits=4):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.quantized = quantized
        self.quantization_bits = quantization_bits
        self._load()
        self._build_pipeline(pipeline_kwargs or {})

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.quantized:
            if self.quantization_bits == 4:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization_bits == 8:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            else:
                raise ValueError("Quantization bits must be 4 or 8.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()

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

    def __call__(self, prompt: str, **generate_kwargs):
        if self.pipe is None:
            raise ValueError("Pipeline not initialized.")
        result = self.pipe(prompt, **generate_kwargs)
        response = result[0]['generated_text']
        return response

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_pipeline(self):
        return self.pipe
