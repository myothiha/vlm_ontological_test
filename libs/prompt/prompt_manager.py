# prompting/manager.py

import os

class PromptTemplateManager:
    def __init__(self, prompt_dir="templates"):
        self.prompt_dir = prompt_dir
        self.templates = self._load_all_prompts()

    def _load_all_prompts(self):
        templates = {}
        for filename in os.listdir(self.prompt_dir):
            if filename.endswith((".txt", ".md")):
                name = os.path.splitext(filename)[0]
                with open(os.path.join(self.prompt_dir, filename), "r", encoding="utf-8") as f:
                    templates[name] = f.read().strip()
        return templates

    def format(self, name, **kwargs):
        if name not in self.templates:
            raise ValueError(f"Prompt template '{name}' not found.")
        return self.templates[name].format(**kwargs)

    def list_templates(self):
        return list(self.templates.keys())

    def get_template(self, name):
        return self.templates.get(name)


# ------------------------------------------
# Usage Example: Test this file standalone
# ------------------------------------------
if __name__ == "__main__":
    print("🔧 Running PromptTemplateManager demo...")

    manager = PromptTemplateManager(prompt_dir="prompt/templates")
    print("Available templates:", manager.list_templates())

    # Use your ontology prompt as an example
    prompt = manager.format("generate_knowledge_questions", class_name="Person")
    print("\n📝 Formatted Prompt:\n")
    print(prompt)