import os
import sys
import csv
import json
import labelbox as lb
from dotenv import load_dotenv

load_dotenv()

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.utils import clean_special_chars

# --- CONFIG ---
PROJECT_UID = os.getenv("label_box_project_id")
OUTPUT_CSV  = "results/00_labelbox_text_concepts_metadata.csv"
NER_TOOL_NAME = "Concept"  # Must match tool name in your ontology
# -------------

client = lb.Client(api_key=os.getenv("label_box_api_key"))
project = client.get_project(PROJECT_UID)

# Export annotations with full details
export_task = project.export(params={
    "attachments": False,
    "metadata_fields": False,
    "data_row_details": True,
    "project_details": True,
    "label_details": True,
    "performance_details": False,
    "interpolated_frames": False
})
export_task.wait_till_done()

rows_out = []
for row in export_task.get_buffered_stream(lb.StreamType.RESULT):
    row_json = row.json

    text_data = row_json["data_row"]["row_data"]
    annotations = row_json.get("annotations", [])

    # Extract concept spans
    concepts = []
    # Navigate into the nested structure
    for project_id, project_data in row_json.get("projects", {}).items():
        for label in project_data.get("labels", []):
            for ann in label.get("annotations", {}).get("objects", []):
                if ann.get("annotation_kind") == "TextEntity" and ann.get("name") == NER_TOOL_NAME:
                    start = ann["location"]["start"]
                    end = ann["location"]["end"]
                    span_text = ann["location"].get("token", text_data[start:end])
                    clean_text = clean_special_chars(span_text.strip())
                    if clean_text:
                        concepts.append(clean_text)

    # Deduplicate
    seen = set()
    concepts_unique = [c for c in concepts if not (c.lower() in seen or seen.add(c.lower()))]

    if not concepts_unique:
        continue

    # Prepare metadata = everything else
    metadata_dict = dict(row_json)  # shallow copy
    metadata_dict.pop("annotations", None)
    metadata_dict.pop("data_row", None)
    metadata_str = json.dumps(metadata_dict, ensure_ascii=False)

    rows_out.append({
        "text": text_data,
        "concepts": json.dumps(concepts_unique, ensure_ascii=False),
        "metadata": metadata_str
    })

# Save CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "concepts", "metadata"])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Exported {len(rows_out)} rows to {OUTPUT_CSV}")