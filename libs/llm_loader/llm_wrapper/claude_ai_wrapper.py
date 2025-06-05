# llm_loader/llm_wrapper.py

import os
import anthropic

class ClaudeAIWrapper:
    def __init__(self, model_name="claude-3-5-sonnet-20240620", temperature=0.2):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.temperature = temperature

    def _format_prompt(self, prompt):
        """
        Accepts either a raw string (user message) or full role-formatted list.
        Returns a list of messages.
        """
        if isinstance(prompt, str):
            return [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
        elif isinstance(prompt, list):
            return prompt
        else:
            raise ValueError("Prompt must be a string or list of messages.")

    def __call__(self, prompt, **kwargs):
        messages = self._format_prompt(prompt)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_new_tokens", 400),
            system="You are a helpful and expert Python programmer.",
            messages=messages
        )

        return response.content[0].text

    def get_model_name(self):
        return self.model_name