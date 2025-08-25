import os, csv, ast, re, uuid
import labelbox as lb
import labelbox.types as lb_types
from dotenv import load_dotenv

load_dotenv()

# ========= CONFIG =========
CSV_PATH          = "results/00_concept_extraction.csv"   # columns: text, concepts
DATASET_UID       = os.getenv("label_box_dataset_id")
PROJECT_UID       = os.getenv("label_box_project_id")
NER_TOOL_NAME     = "Concept"    # EXACT name of your NER tool in ontology
CREATE_DATA_ROWS  = False         # set False if rows already uploaded
WORD_BOUNDARY     = False        # True = match whole words only
# ==========================

def parse_concepts(cell):
    if cell is None or str(cell).strip() == "":
        return []
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]

def find_all_spans(text, phrase, word_boundary=False):
    if not phrase:
        return []
    esc = re.escape(phrase)
    pattern = rf"\b{esc}\b" if word_boundary else esc
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]

load_dotenv()
api_key = os.getenv("label_box_api_key")
assert api_key, "Missing env var 'label_box_api_key'"

client  = lb.Client(api_key=api_key)

# Project & ontology
project  = client.get_project(PROJECT_UID)
ontology = project.ontology()
tool_names = [t.get("name") for t in ontology.normalized.get("tools", [])]
assert NER_TOOL_NAME in tool_names, f"NER tool '{NER_TOOL_NAME}' not found. Available: {tool_names}"

# Dataset
dataset = client.get_dataset(DATASET_UID)

# Read CSV rows
rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    assert "text" in reader.fieldnames and "concepts" in reader.fieldnames, "CSV needs 'text' and 'concepts'"
    for i, r in enumerate(reader):
        rows.append({
            "gk":   (r.get("id") or f"text_{i}").strip(),
            "text": (r.get("text") or "").strip(),
            "concepts_raw": r.get("concepts")
        })

# (A) Upload data rows (optional)
if CREATE_DATA_ROWS:
    assets = []
    for r in rows:
        attachments = []
        if r["concepts_raw"]:
            attachments.append({"type": "RAW_TEXT", "value": f"concepts: {r['concepts_raw']}"})
        assets.append({
            "row_data": r["text"],
            "global_key": r["gk"],
            "media_type": "TEXT",
            "attachments": attachments
        })
    task = dataset.create_data_rows(assets)
    task.wait_till_done()
    print("Upload errors:", task.errors)
    print(f"Uploaded {len(assets)} rows.")
# (per docs, rows must be in Catalog before attaching annotations). :contentReference[oaicite:1]{index=1}

# (B) Build Python Annotation Types for NER entities
labels = []
for r in rows:
    text = r["text"]
    entities = []
    for concept in parse_concepts(r["concepts_raw"]):
        for (start, end) in find_all_spans(text, concept, WORD_BOUNDARY):
            ent = lb_types.TextEntity(start=start, end=end)
            entities.append(lb_types.ObjectAnnotation(name=NER_TOOL_NAME, value=ent))
    if entities:
        labels.append(lb_types.Label(
            data={"global_key": r["gk"]},
            annotations=entities,
            # is_benchmark_reference=True  # only for ground-truth import, not MAL
        ))

print(f"Prepared {len(labels)} label objects with NER entities.")

# (C) Import as prelabels (MAL) per docs
# Option A: prelabels / model-assisted labeling (attach predictions to project) :contentReference[oaicite:2]{index=2}
if labels:
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=project.uid,
        name="mal_job_" + str(uuid.uuid4()),
        predictions=labels
    )
    print("MAL import status:", upload_job.statuses)
    print("MAL import errors:", upload_job.errors)
else:
    print("No entities to import (no matches found).")

# If you want to import as ground truth instead, use:
# lb.LabelImport.create_from_objects(client=client, project_id=project.uid, name="gt_job_"+str(uuid.uuid4()), labels=labels)
# (Relationships are not supported in ground-truth import). :contentReference[oaicite:3]{index=3}
