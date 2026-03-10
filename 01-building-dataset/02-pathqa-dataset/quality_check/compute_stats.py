"""
Quality Statistics – VLM Reasoning Questions (Multi-Label)
===========================================================
Reads data/labels.json (from the labeling app) and computes per-concept
quality counts across three issue dimensions:

  • incoherence   – question logic is flawed
  • misalignment  – question is not relevant to the concept / image
  • language-bias – question can be answered from text alone without the image

A question is "good" if it carries NONE of the three issue labels.
A question can have multiple issues simultaneously (counts are not exclusive).

Verdict (per concept)
---------------------
  PASS  if  good_questions / total_questions  ≥  threshold  (default 80 %)
  FAIL  otherwise
  NOT STARTED  if no questions have been labeled for the concept

Outputs
-------
  data/quality_stats.json   – machine-readable full results
  data/quality_stats.csv    – spreadsheet-friendly table

Usage
-----
    conda run -n myo_thesis python compute_stats.py
    conda run -n myo_thesis python compute_stats.py --threshold 80
    conda run -n myo_thesis python compute_stats.py --show-not-started
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
KB_DIR      = BASE_DIR.parent / "results" / "knowledge_base"
DATA_DIR    = BASE_DIR / "data"
LABELS_FILE = DATA_DIR / "labels.json"
MANIFEST    = DATA_DIR / "sampled_concepts.json"
OUT_JSON    = DATA_DIR / "quality_stats.json"
OUT_CSV     = DATA_DIR / "quality_stats.csv"

DIMENSION_FILES = [
    ("properties.csv",           "Properties"),
    ("functions.csv",            "Functions"),
    ("relational.csv",           "Relational"),
    ("contexual_properties.csv", "Contextual Properties"),
    ("contexual_behavior.csv",   "Contextual Behavior"),
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_questions(csv_path: Path) -> list:
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


def compute_concept(concept: str, concept_labels: dict) -> dict:
    """Scan KB files and compute issue counts for one concept."""
    total = flagged = incoherence = misalignment = language_bias = 0

    for fname, dname in DIMENSION_FILES:
        fpath = KB_DIR / concept / "vlm_reasoning_questions" / fname
        dim_labels = concept_labels.get(dname, {})

        for q in read_questions(fpath):
            total += 1
            raw = dim_labels.get(q) or []
            if isinstance(raw, str):   # backward-compat with old good/bad format
                raw = []
            issues = [x for x in raw if x in ("incoherence", "misalignment", "language-bias")]
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


# ─── Core ────────────────────────────────────────────────────────────────────

def compute(threshold: float) -> dict:
    # Load labels
    if not LABELS_FILE.exists():
        print("ERROR: data/labels.json not found. Label some questions first.")
        sys.exit(1)
    labels: dict = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    # Load concept list
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        concepts = manifest.get("concepts", [])
    else:
        concepts = sorted(labels.keys())
        print("[WARN] Manifest not found; using concepts present in labels.json.")

    # Per-concept stats
    rows = []
    for concept in concepts:
        c = compute_concept(concept, labels.get(concept, {}))

        started = c["flagged"] > 0  # at least one issue was tagged
        if c["total"] == 0:
            proportion = None
            verdict    = "NO QUESTIONS"
        elif not started:
            proportion = 1.0          # no issues found → 100% good
            verdict    = "NOT STARTED"
        else:
            proportion = c["good"] / c["total"]
            verdict    = "PASS" if proportion * 100 >= threshold else "FAIL"

        rows.append({
            "concept":            concept,
            "display_name":       concept.replace("_", " ").title(),
            "total":              c["total"],
            "good":               c["good"],
            "flagged":            c["flagged"],
            "incoherence":          c["incoherence"],
            "misalignment":         c["misalignment"],
            "language_bias":        c["language_bias"],
            "good_pct":           round(proportion * 100, 1) if proportion is not None else None,
            "verdict":            verdict,
        })

    # Summary (exclude NOT STARTED from PASS/FAIL counts)
    started_rows = [r for r in rows if r["verdict"] not in ("NOT STARTED", "NO QUESTIONS")]
    summary = {
        "threshold_pct":      threshold,
        "total_concepts":     len(rows),
        "started_concepts":   len(started_rows),
        "not_started":        len(rows) - len(started_rows),
        "pass_concepts":      sum(1 for r in started_rows if r["verdict"] == "PASS"),
        "fail_concepts":      sum(1 for r in started_rows if r["verdict"] == "FAIL"),
        "total_questions":    sum(r["total"]      for r in rows),
        "total_good":         sum(r["good"]       for r in started_rows),
        "total_flagged":      sum(r["flagged"]    for r in rows),
        "total_incoherence":    sum(r["incoherence"]  for r in rows),
        "total_misalignment":   sum(r["misalignment"] for r in rows),
        "total_language_bias":  sum(r["language_bias"]for r in rows),
    }

    return {"summary": summary, "concepts": rows}


def save_results(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV
    fields = [
        "concept", "display_name", "total", "good", "flagged",
        "incoherence", "misalignment", "language_bias", "good_pct", "verdict",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["concepts"])


def print_report(data: dict, show_not_started: bool) -> None:
    s    = data["summary"]
    rows = data["concepts"]

    print(f"\n{'═'*76}")
    print(f"  VLM Reasoning Question Quality Report")
    print(f"  Threshold : ≥ {s['threshold_pct']:.0f}% good  →  PASS")
    print(f"{'═'*76}")
    print(f"  Total concepts sampled   : {s['total_concepts']}")
    print(f"  Concepts reviewed        : {s['started_concepts']}")
    print(f"  Not yet started          : {s['not_started']}")
    print(f"  PASS                     : {s['pass_concepts']}")
    print(f"  FAIL                     : {s['fail_concepts']}")
    print(f"{'─'*76}")
    print(f"  Total questions          : {s['total_questions']}")
    print(f"  Good (no issues)         : {s['total_good']}")
    print(f"  Flagged (any issue)      : {s['total_flagged']}")
    print(f"  └─ Incoherence             : {s['total_incoherence']}")
    print(f"  └─ Misalignment            : {s['total_misalignment']}")
    print(f"  └─ Language Bias           : {s['total_language_bias']}")
    print(f"  (issue counts are NOT exclusive — one question can have multiple)")
    print(f"{'═'*76}\n")

    col = 45
    print(f"  {'Concept':<{col}}  {'Good':>5} {'Flagged':>7} {'Incoh':>5} {'Misal':>5} {'LangB':>5}  {'%Good':>7}  Verdict")
    print(f"  {'─'*76}")

    ORDER = {"FAIL": 0, "PASS": 1, "NOT STARTED": 2, "NO QUESTIONS": 3}
    sorted_rows = sorted(rows, key=lambda r: (ORDER.get(r["verdict"], 9), r["display_name"]))

    for r in sorted_rows:
        if not show_not_started and r["verdict"] in ("NOT STARTED", "NO QUESTIONS"):
            continue
        pct     = f"{r['good_pct']:.1f}%" if r["good_pct"] is not None else "  N/A"
        verdict = r["verdict"]
        icon    = {"PASS": "✅", "FAIL": "❌", "NOT STARTED": "⬜", "NO QUESTIONS": "—"}.get(verdict, "")
        print(f"  {r['display_name']:<{col}}  {r['good']:>5} {r['flagged']:>7} "
              f"{r['incoherence']:>5} {r['misalignment']:>5} {r['language_bias']:>5}  "
              f"{pct:>7}  {icon} {verdict}")

    print(f"\n  Saved → {OUT_JSON}")
    print(f"          {OUT_CSV}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quality statistics from multi-label VLM reasoning question reviews."
    )
    parser.add_argument(
        "--threshold", type=float, default=80.0,
        help="Minimum %% of good (unlabeled) questions for a concept to PASS (default: 80)."
    )
    parser.add_argument(
        "--show-not-started", action="store_true",
        help="Include concepts that haven't been reviewed yet in the console table."
    )
    args = parser.parse_args()

    data = compute(args.threshold)
    save_results(data)
    print_report(data, args.show_not_started)


if __name__ == "__main__":
    main()
