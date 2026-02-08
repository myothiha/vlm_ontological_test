# Ollama Container Setup and Wrapper Class

- [Files Structure](#files-structure)
- [Setup Ollama Container](#how-to-run)
- [Manage Models](#manage-models)
- [OllamaWrapper Class Usage](#ollamawrapper-class-usage)

## Files Structure
 - `llm_loader/ollama/ollama_wrapper.py` contains the wrapper class used to interact with the Ollama models.
 - `Dockerfile` defines the environment setup for running Ollama in a container.
 - `docker-compose.yaml` is used to orchestrate the Ollama container and manage its dependencies.

## Setup Ollama Container
 - Create a new folder called `ollama-api` and copy `docker-compose.yaml` to  that folder. 
 - Run `cd ollama-api`.
 - Run `docker-compose up` in the folder.
 - Once running, Ollama service will be available at [http://localhost:11434](http://localhost:11434).
 - Use the wrapper class in `ollama_wrapper.py` to send requests to the running Ollama container.

## Manage Models
 - To List available models inside the Ollama container: `docker exec -it ollama ollama list`.
 - To pull a new model (example: LLM3 8B model): `docker exec -it ollama ollama list`

## OllamaWrapper Class Usage

### For text only LLM
 - We can use text only LLM for single turn or multi-turns depending on the use case. By default, it is single turn. 
 - Example usage (single turn):
   ```python
   from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

   llm = OllamaWrapper(model="gpt-oss:20b")
   response = llm("Hello, world!", max_new_tokens=128)
   print(response)
   ```
 - Example usage (Multi turn):
   ```python
   from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

   llm = OllamaWrapper(model="gpt-oss:20b", multi_turn=True)
   ollama.set_system_prompt("You are a helpful assistant.") # only for multi-turn mode

   response = llm("What is the capital of Paris?", max_new_tokens=128)
   print(response)

   response = llm("What is its population.", max_new_tokens=128)
   print(response)

   # Print the history
   print(llm.history)
   ```

### For LLM with image input.
 - We can use LLM wrapper with image input. Right now it's single turn only.
 - Example usage (single turn):
   ```python
   from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

   llm = OllamaWrapper(model="llava:34b")
   response = llm(images=["example.jpg"], prompt="Describe the provided image", max_new_tokens=128)
   print(response)
   ```