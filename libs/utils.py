import re
import json

def extract_list(response):
    # Remove prompt if present (everything before 'Output: [')
    output_start = response.find('<begin>')
    if output_start != -1:
        response = response[output_start + len('Output: '):].strip()

    print("Responose:", response)
    # Find all JSON arrays in the string
    arrays = re.findall(r"\[\s*\".*?\"\s*\]", response, re.DOTALL)

    if arrays:
        # Get the last array (the one for "Person")
        person_json = arrays[-1]
        try:
            questions = json.loads(person_json)
            return questions
        except json.JSONDecodeError as e:
            return "Error: Failed to parse extracted JSON."
    else:
        # Try to find an empty list
        empty_list_match = re.search(r"\[\s*\]", response)
        if empty_list_match:
            return []
        return "Error: No JSON array found."

def extract_json_object(response):
    """
    Extract and parse the first JSON object found in the response.
    """
    try:
        # Find the first complete JSON object in the string
        match = re.search(r"\{\s*.*?\s*\}", response, re.DOTALL)
        if match:
            obj_text = match.group()
            obj = json.loads(obj_text)
            return obj
        else:
            print("No JSON object found.")
            return None
    except json.JSONDecodeError as e:
        print("❌ Failed to parse extracted JSON object:", e)
        return None
