import re
import json

def extract_list1(response):
    # Remove everything before the first 'Output: ['
    output_idx = response.find('Output: [')
    if output_idx != -1:
        response = response[output_idx + len('Output: '):].strip()
    else:
        bracket_idx = response.find('[')
        if bracket_idx != -1:
            response = response[bracket_idx:]
    # Find all JSON arrays in the string
    arrays = re.findall(r"\[\s*\".*?\"\s*\]", response, re.DOTALL)
    if arrays:
        array_json = arrays[0]
        try:
            extracted_list = json.loads(array_json)
            return extracted_list
        except json.JSONDecodeError as e:
            return "Error: Failed to parse extracted JSON."
    else:
        # Try to find an empty list
        empty_list_match = re.search(r"\[\s*\]", response)
        if empty_list_match:
            return []
        return "Error: No JSON array found."
    
def extract_list_from_gpt(response):
    # Find all JSON arrays in the string
    arrays = re.findall(r"\[\s*\".*?\"\s*\]", response, re.DOTALL)

    if arrays:
        # Get the last array (the one for "Person")
        person_json = arrays[0]
        try:
            questions = json.loads(person_json)
            return questions
        except json.JSONDecodeError as e:
            print("Error: Failed to parse extracted JSON:")
            return "Error: Failed to parse extracted JSON:"
    else:
        print("Error: No JSON array found.")
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

def extract_list(response):
    """
    Extract the first JSON array that appears after the first '<begin>' in the response.
    This avoids regex and uses string manipulation + json.loads for robustness.
    Returns the parsed list, or an error string if extraction fails.
    """
    begin_idx = response.find('<begin>')
    if begin_idx == -1:
        print("Error: '<begin>' not found in the response.")
        return []
    response = response[begin_idx + len('<begin>'):].strip()
    
    output_idx = response.find('Output:')
    if output_idx == -1:
        print("Error: 'Output:' not found in the response.")
        return []
    # Find the first '[' after 'Output:'
    json_start = response.find('[', output_idx)
    if json_start == -1:
        print("Error: No JSON array found after 'Output:'.")
        return []
    # Find the matching closing ']' for the array
    bracket_count = 0
    for i in range(json_start, len(response)):
        if response[i] == '[':
            bracket_count += 1
        elif response[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                json_end = i + 1
                break
    else:
        print("Error: No matching closing ']' found.")
        return []
    array_str = response[json_start:json_end]
    try:
        return json.loads(array_str)
    except Exception as e:
        print(f"Error: Failed to parse extracted JSON array: {e}")
        return []
