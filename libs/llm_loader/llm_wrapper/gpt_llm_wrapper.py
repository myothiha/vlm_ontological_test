# llm_loader/llm_wrapper.py

import os
from openai import OpenAI
from dotenv import load_dotenv

class GPTLLMWrapper:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.2):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY is not set in the environment.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature

    def _format_prompt(self, prompt):
        """
        Accepts either a raw string (user message) or full role-formatted list.
        Returns a list of messages.
        """
        if isinstance(prompt, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        elif isinstance(prompt, list):
            return prompt
        else:
            raise ValueError("Prompt must be a string or list of messages.")

    def __call__(self, prompt, **kwargs):
        messages = self._format_prompt(prompt)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_new_tokens", 500),
            top_p=kwargs.get("top_p", 1.0),
        )

        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_name
