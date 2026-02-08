import requests
import json
from dotenv import load_dotenv
import os
from PIL import Image, ImageOps
import io
import base64
from pathlib import Path
from typing import Iterable, List, Optional, Union

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

    def __call__(self, prompt: str, images = None, stream: bool = False, **kwargs):
        if images is None:
            if self.multi_turn:
                # print("Using multi-turn mode")
                return self.chat(prompt, stream=stream, **kwargs)
            
            # print("Using single-turn mode")
            return self.generate(prompt, stream=stream, **kwargs)
        else:
            print("Using multimodal generation")
            return self.generate_multimodal(
                prompt=prompt,
                images=images,
                stream=stream,
                **kwargs
            )

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
            try:
                return response.json()["response"]
            except Exception as e:
                print("Error in response:", response)
                return ""
        
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
                yield collected
            return stream_generator()
        else:
            result = response.json()["message"]["content"]
            self.history.append({"role": "assistant", "content": result})
            return result
        
    def generate_multimodal(
        self,
        prompt: str,
        images: Union[str, Path, bytes, Iterable[Union[str, Path, bytes]]],
        *,
        stream: bool = False,
        keep_alive: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        def _pil_to_bytes(img: Image.Image) -> bytes:
            # Fix orientation from EXIF (avoids rotated uploads)
            img = ImageOps.exif_transpose(img)

            # If image has alpha, keep PNG; otherwise use JPEG (smaller)
            has_alpha = img.mode in ("RGBA", "LA") or ("transparency" in img.info)

            # Normalize mode: CMYK/YCbCr/P/L/... → RGB; keep RGBA if alpha
            if has_alpha and img.mode != "RGBA":
                img = img.convert("RGBA")
            elif not has_alpha and img.mode != "RGB":
                img = img.convert("RGB")

            buf = io.BytesIO()
            if has_alpha:
                # PNG supports alpha
                img.save(buf, format="PNG", optimize=True)
            else:
                # JPEG (no alpha) — smaller payloads for Ollama
                img.save(buf, format="JPEG", quality=90, optimize=True)
            return buf.getvalue()
        
        def _to_b64(img) -> str:
            if isinstance(img, (str, Path)):
                p = Path(img)
                if not p.exists():
                    raise FileNotFoundError(f"Image not found: {p}")
                data = p.read_bytes()

            elif isinstance(img, (bytes, bytearray)):
                data = bytes(img)

            elif isinstance(img, Image.Image):  # PIL image
                data = _pil_to_bytes(img)

            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            return base64.b64encode(data).decode("utf-8")
    
        # Normalize images -> List[base64 str]
        if isinstance(images, (str, Path, bytes, bytearray)):
            b64_images: List[str] = [_to_b64(images)]
        else:
            b64_images = [_to_b64(i) for i in images]

        if not b64_images:
            raise ValueError("At least one image must be provided.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": b64_images,
            "stream": stream,
            **kwargs
        }
        # if options:
        #     payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        url = f"{self.base_url.rstrip('/')}/api/generate"

        # non-streaming
        if not stream:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "")

        # streaming
        response = requests.post(url, json=payload, stream=True, timeout=timeout)
        response.raise_for_status()

        def _iter():
            import json
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunk = obj.get("response", "")
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        break
                except Exception:
                    continue

        return _iter()

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

if __name__ == "__main__":
    # Example usage
    ollama = OllamaWrapper(model="llava:34b", multi_turn=True)
    ollama.set_system_prompt("You are a helpful assistant.") # only for multi-turn mode
    
    print("Single-turn generation:")
    response = ollama.generate("What is the capital of France?", temperature=0.1)
    print(response)
    
    print("\nMulti-turn chat:")
    response = ollama.chat("Hello, who won the World Cup in 2018?", temperature=0.1)
    print(response)
    response = ollama.chat("Where was it held?", temperature=0.1)
    print(response)
    
    print("\nMultimodal generation:")
    multimodal_response = ollama.generate_multimodal(
        prompt="Describe the main object and its condition.",
        images=["example.jpg"],
        options={"temperature": 0.1}
    )
    print(multimodal_response)