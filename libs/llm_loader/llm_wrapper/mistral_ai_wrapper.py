import os
from mistralai import Mistral

class MistralAIWrapper:
    def __init__(self, model_name="mistral-large-latest", temperature=0):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set in the environment.")
        self.model_name = model_name
        self.temperature = temperature
        self.client = Mistral(api_key=self.api_key)

    def _format_prompt(self, prompt):
        """
        Accepts either a raw string (user message) or a list of messages.
        Returns a list of messages.
        """
        if isinstance(prompt, str):
            return [
                {"role": "user", "content": prompt}
            ]
        elif isinstance(prompt, list):
            return prompt
        else:
            raise ValueError("Prompt must be a string or list of messages.")

    def __call__(self, prompt, **kwargs):
        messages = self._format_prompt(prompt)
        response = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_new_tokens", 500),
            top_p=kwargs.get("top_p", 1),
        )
        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_name
