"""
VLM Reasoning Questions – Labeling Web App
===========================================
Multi-label quality tagging: each question can be tagged with any combination of
  incoherence | misalignment | language-bias
Questions with NO tags are implicitly "good".

Run:
    conda run -n myo_thesis python labeling_app.py
    Open http://localhost:5050
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request, Response

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
KB_DIR      = BASE_DIR.parent / "results" / "knowledge_base"
DATA_DIR    = BASE_DIR / "data"
LABELS_FILE = DATA_DIR / "labels.json"
MANIFEST    = DATA_DIR / "sampled_concepts.json"

DIMENSION_FILES = [
    ("properties.csv",           "Properties"),
    ("functions.csv",            "Functions"),
    ("relational.csv",           "Relational"),
    ("contexual_properties.csv", "Contextual Properties"),
    ("contexual_behavior.csv",   "Contextual Behavior"),
]

VALID_ISSUES = {"incoherence", "misalignment", "language-bias"}

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── Authentication ────────────────────────────────────────────────────────────

AUTH_USER = "admin"
AUTH_PASS = "T8v!mK9$qL2#pR5_"

def check_auth(username, password):
    return username == AUTH_USER and password == AUTH_PASS

def authenticate():
    return Response(
        "Unauthorized Access", 401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

@app.before_request
def require_auth():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_labels() -> Dict:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return {}


def save_labels(labels: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_manifest() -> Optional[List[str]]:
    if not MANIFEST.exists():
        return None
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data.get("concepts", [])


def read_questions(csv_path: Path) -> List[str]:
    questions = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("content", "").strip()
                if q:
                    questions.append(q)
    except Exception:
        pass
    return questions


def concept_stats(concept: str, labels: Dict) -> Dict:
    """
    Return per-concept quality counts.
    labels[concept][dim][question] = ["incoherence", "language-bias"]  (list of issues)
    Questions absent from labels → no issues → good.
    """
    concept_labels = labels.get(concept, {})
    total = flagged = incoherence = misalignment = language_bias = 0

    for fname, dname in DIMENSION_FILES:
        fpath = KB_DIR / concept / "vlm_reasoning_questions" / fname
        for q in read_questions(fpath):
            total += 1
            issues = concept_labels.get(dname, {}).get(q) or []
            if isinstance(issues, str):          # migrate old "good"/"bad" format
                issues = []
            if issues:
                flagged += 1
                if "incoherence"   in issues: incoherence  += 1
                if "misalignment"  in issues: misalignment += 1
                if "language-bias" in issues: language_bias += 1

    return {
        "total":      total,
        "flagged":    flagged,
        "good":       total - flagged,
        "incoherence":  incoherence,
        "misalignment": misalignment,
        "language_bias":language_bias,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/concepts")
def api_concepts():
    concepts = load_manifest()
    if concepts is None:
        return jsonify({"error": "Manifest not found. Run sample_vlm_reasoning_questions.py first."}), 404

    labels = load_labels()
    result = []
    for name in concepts:
        stats = concept_stats(name, labels)
        result.append({
            "name":         name,
            "display_name": name.replace("_", " ").title(),
            **stats,
        })
    return jsonify(result)


@app.route("/api/concept/<name>")
def api_concept(name: str):
    labels = load_labels()
    concept_labels = labels.get(name, {})

    dimensions = []
    for fname, dname in DIMENSION_FILES:
        fpath = KB_DIR / name / "vlm_reasoning_questions" / fname
        qs = read_questions(fpath)
        if qs:
            dim_labels = concept_labels.get(dname, {})
            questions  = []
            for q in qs:
                raw = dim_labels.get(q) or []
                if isinstance(raw, str):     # migrate old format gracefully
                    raw = []
                questions.append({"text": q, "labels": raw})
            dimensions.append({"name": dname, "questions": questions})

    return jsonify({
        "name":         name,
        "display_name": name.replace("_", " ").title(),
        "dimensions":   dimensions,
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    """
    Body: { concept, dimension, question, labels: ["incoherence", ...] }
    labels is a list of active issue strings.  Empty list = no issues (good).
    """
    data      = request.get_json(force=True)
    concept   = data["concept"]
    dim       = data["dimension"]
    q         = data["question"]
    issues    = [x for x in (data.get("labels") or []) if x in VALID_ISSUES]

    labels = load_labels()
    labels.setdefault(concept, {}).setdefault(dim, {})

    if issues:
        labels[concept][dim][q] = issues
    else:
        # No issues → remove from store (implicitly good)
        labels[concept][dim].pop(q, None)

    save_labels(labels)
    return jsonify({"status": "ok"})


@app.route("/api/progress")
def api_progress():
    """Global progress: total questions, flagged questions, concepts with ≥1 flag."""
    concepts = load_manifest() or []
    labels   = load_labels()
    total = flagged = concepts_flagged = 0
    for c in concepts:
        s = concept_stats(c, labels)
        total   += s["total"]
        flagged += s["flagged"]
        if s["flagged"] > 0:
            concepts_flagged += 1
    return jsonify({
        "total":            total,
        "flagged":          flagged,
        "concepts_flagged": concepts_flagged,
        "total_concepts":   len(concepts),
    })


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not MANIFEST.exists():
        print(
            "ERROR: data/sampled_concepts.json not found.\n"
            "Run sample_vlm_reasoning_questions.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Starting VLM Labeling App → http://localhost:5050")
    app.run(debug=False, port=5050, host="0.0.0.0")
