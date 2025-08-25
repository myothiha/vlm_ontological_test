import csv
import json
import re

INPUT_CSV = "../results/00_concept_extraction.csv"   # your dataset
OUTPUT_JSONL = "results/doccano_dataset.jsonl"

def replace_special_chars(text: str) -> str:
    """Clean text from special characters if needed."""
    return text.replace("\n", " ").replace("\r", " ").strip()

def find_all_occurrences(text: str, sub: str):
    """Find all start/end indices of substring in text (case-insensitive)."""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(sub), text, re.IGNORECASE)]

def csv_to_doccano(input_csv, output_jsonl):
    with open(input_csv, newline="", encoding="utf-8") as f, \
         open(output_jsonl, "w", encoding="utf-8") as out_f:

        reader = csv.DictReader(f)
        for row in reader:
            text = replace_special_chars(row["text"])
            concepts_raw = row.get("concepts", "").strip()

            labels = []
            if concepts_raw:
                try:
                    # Convert string list into Python list
                    concepts = eval(concepts_raw) if concepts_raw.startswith("[") else [concepts_raw]
                    for concept in concepts:
                        concept = concept.strip()
                        if not concept:
                            continue
                        occurrences = find_all_occurrences(text, concept)
                        if occurrences:
                            for start, end in occurrences:
                                labels.append([start, end, "CONCEPT"])
                        else:
                            # Concept not found in text → keep it anyway
                            labels.append([0, 0, "CONCEPT"])
                except Exception as e:
                    print(f"Skipping row due to error: {e}")

            # Save even if labels are empty (to preserve row)
            json_line = {"text": text, "label": labels}
            out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"✅ Saved dataset in Doccano format at {output_jsonl}")

# Run conversion
csv_to_doccano(INPUT_CSV, OUTPUT_JSONL)
