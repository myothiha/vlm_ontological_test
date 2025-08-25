import requests
import json
from dotenv import load_dotenv
import os
class OllamaWrapper:
    def __init__(self, model: str, multi_turn: bool = False):
        load_dotenv()
        
        host_url = os.getenv("ollama_host")

        if not host_url:
            raise ValueError("Environment variable 'ollama_host' is not set. Please set it to the Ollama server URL.")

        self.base_url = host_url.rstrip("/")
        self.model = model
        self.multi_turn = multi_turn
        self.system_prompt = None
        self.history = []

    def set_system_prompt(self, prompt: str):
        if self.multi_turn:
            self.system_prompt = prompt
            self.history.append({"role": "system", "content": self.system_prompt})
        else:
            raise ValueError("System prompt can only be set in multi-turn mode.")

    def set_mode(self, multi_turn: bool):
        self.multi_turn = multi_turn

    def __call__(self, prompt: str, stream: bool = False, **kwargs):
        if self.multi_turn:
            # print("Using multi-turn mode")
            return self.chat(prompt, stream=stream, **kwargs)
        
        # print("Using single-turn mode")
        return self.generate(prompt, stream=stream, **kwargs)

    def generate(self, prompt: str, stream: bool = False, **kwargs):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=payload, stream=stream)

        if stream:
            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)["response"]
            return stream_generator()
        else:
            return response.json()["response"]
        
    def chat(self, prompt: str, stream: bool = False, **kwargs):

        url = f"{self.base_url}/api/chat"

        self.history.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=payload, stream=stream)

        if stream:
            def stream_generator():
                collected = ""
                for line in response.iter_lines():
                    if line:
                        content = json.loads(line)["message"]["content"]
                        print(content, end='', flush=True)
                        collected += content
                self.history.append({"role": "assistant", "content": collected})
                print()
                yield collected
            return stream_generator()
        else:
            result = response.json()["message"]["content"]
            self.history.append({"role": "assistant", "content": result})
            return result

    # def chat(self, messages: list, stream: bool = False, **kwargs):
    #     url = f"{self.base_url}/api/chat"
    #     payload = {
    #         "model": self.model,
    #         "messages": messages,
    #         "stream": stream,
    #         **kwargs
    #     }
    #     response = requests.post(url, json=payload, stream=stream)
    #     if stream:
    #         def stream_generator():
    #             for line in response.iter_lines():
    #                 if line:
    #                     yield json.loads(line)["message"]["content"]
    #         return stream_generator()
    #     else:
    #         return response.json()["message"]["content"]

    def pull(self, model: str):
        url = f"{self.base_url}/api/pull"
        payload = {"name": model}
        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                print(json.loads(line))

    def list_models(self):
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return response.json().get("models", [])

    def delete(self, model: str):
        url = f"{self.base_url}/api/delete"
        payload = {"name": model}
        response = requests.delete(url, json=payload)
        return response.ok
