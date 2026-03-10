# Quality Check – VLM Reasoning Questions

This folder provides a **three-step quality-check workflow** for the
`vlm_reasoning_questions` generated for each concept in the PathQA
knowledge base (`results/knowledge_base/`).

---

## Background

The knowledge base contains **~541 concept directories**, each with a
`vlm_reasoning_questions/` subfolder holding five dimension-specific CSV files:

| CSV file | Dimension |
|---|---|
| `properties.csv` | Physical / observable properties |
| `functions.csv` | Functional roles and behaviours |
| `relational.csv` | Relations to anatomical structures or diseases |
| `contexual_properties.csv` | Context-dependent property changes |
| `contexual_behavior.csv` | Context-dependent behavioural changes |

Reviewing all 500+ concepts manually is not feasible, so the workflow
**randomly samples** a subset, presents questions in an interactive browser
UI for labeling, then computes per-concept quality statistics.

---

## Directory Structure

```
quality_check/
├── README.md                          ← you are here
├── sample_vlm_reasoning_questions.py  ← Step 1: sample concepts
├── labeling_app.py                    ← Step 2: web labeling app (Flask)
├── compute_stats.py                   ← Step 3: quality statistics
├── templates/
│   └── index.html                     ← labeling UI (used by Flask)
├── data/                              ← auto-created at runtime
│   ├── sampled_concepts.json          ← manifest of sampled concept names
│   ├── labels.json                    ← all Good/Bad labels (persisted)
│   ├── quality_stats.json             ← computed results (JSON)
│   └── quality_stats.csv             ← computed results (CSV/spreadsheet)
└── outputs/                           ← plain-text previews (10 per file)
    ├── sample_01.txt  (concepts  1–10)
    ├── sample_02.txt  (concepts 11–20)
    ├── sample_03.txt  (concepts 21–30)
    ├── sample_04.txt  (concepts 31–40)
    └── sample_05.txt  (concepts 41–50)
```

---

## Workflow Overview

```
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│  Step 1                  │   │  Step 2                  │   │  Step 3                  │
│  sample_vlm_             │──▶│  labeling_app.py         │──▶│  compute_stats.py        │
│  reasoning_questions.py  │   │  (web app)               │   │                          │
│                          │   │                          │   │  Reads labels.json       │
│  Randomly select N       │   │  Open browser UI         │   │  Computes per-concept    │
│  concepts (seed=42)      │   │  Label each question     │   │  good/bad counts         │
│  Save manifest +         │   │  Good or Bad             │   │  Outputs PASS/FAIL       │
│  preview .txt files      │   │  (auto-saved)            │   │  per concept             │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

---

## Step 1 – Sample Concepts

Randomly pick N valid concepts (those with at least one VLM question),
write preview text files, and save the **manifest** required by the labeling app.

```bash
cd 01-building-dataset/02-pathqa-dataset/quality_check
conda run -n myo_thesis python sample_vlm_reasoning_questions.py
```

### Options

| Argument | Default | Description |
|---|---|---|
| `--n` | `50` | Number of concepts to randomly sample |
| `--seed` | `42` | Random seed — same seed always gives same sample |
| `--per-file` | `10` | Concepts per output preview `.txt` file |

### Outputs

| File | Description |
|---|---|
| `data/sampled_concepts.json` | Manifest of selected concept names (**required by labeling app**) |
| `outputs/sample_01.txt` … | Human-readable question previews |

---

## Step 2 – Label Questions (Web App)

An interactive browser-based UI. Each VLM question can be marked **Good** or
**Bad** with one click. Labels are auto-saved instantly to `data/labels.json`.

### Prerequisites

```bash
# Run once if Flask is not already installed
pip install flask
```

### Start the app

```bash
conda run -n myo_thesis python labeling_app.py
# Then open: http://localhost:5050
```

### UI at a glance

| Area | Description |
|---|---|
| **Header** | Global progress bar — X/Y questions labeled overall |
| **Sidebar** | All 50 sampled concepts with labeled/total badge |
| **Search** | Filter concepts by name in real-time |
| **Main panel** | Questions for selected concept, grouped by dimension |
| **Good / Bad buttons** | Click to assign label; click same button again to **clear** |
| **Colored badges** | Green `✓ Done`, yellow `X/Y` partial, grey `0/N` unlabeled |

### Tips

- **Good** → question is well-formed, realistic, and useful for evaluating VLM understanding.
- **Bad** → question is unclear, irrelevant, or poorly worded.
- Labels survive app restarts — progress is never lost.
- You can label in multiple sessions; just restart the app each time.

---

## Step 3 – Compute Statistics

After labeling (or mid-way through), run the stats script to compute results.

```bash
conda run -n myo_thesis python compute_stats.py
```

### Verdict rule

> A concept **PASSES** if the proportion of `good`-labeled questions ≥ threshold.

```
good_proportion = (good_count / total_labeled) × 100
verdict = PASS  if good_proportion ≥ 80%  (configurable)
          FAIL  otherwise
```

### CLI options

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `80.0` | Minimum % of good questions for PASS |
| `--include-unlabeled` | off | Include unlabeled concepts in output |

### Example console output

```
════════════════════════════════════════════════════════════════════════
  VLM Reasoning Question Quality Report
  Threshold : ≥ 80% good  →  PASS
════════════════════════════════════════════════════════════════════════
  Total concepts sampled  : 50
  Labeled concepts        : 50
  PASS                    : 43
  FAIL                    : 7
  Total questions labeled : 1505
  Good                    : 1278
  Bad                     : 227
════════════════════════════════════════════════════════════════════════

  Concept                                           Good    Bad    %Good  Verdict
  ──────────────────────────────────────────────────────────────────────────────
  Acrocyanosis                                        20      1    95.2%  ✅ PASS
  Adenomyosis                                         25      2    92.6%  ✅ PASS
  Adenosis                                             8     11    42.1%  ❌ FAIL
```

### Output files

| File | Description |
|---|---|
| `data/quality_stats.json` | Full results with summary block (machine-readable) |
| `data/quality_stats.csv` | Per-concept table: concept, good, bad, %, verdict |

---

## Quick-start Cheat Sheet

```bash
cd 01-building-dataset/02-pathqa-dataset/quality_check

# 1. Sample (already done – manifet exists)
conda run -n myo_thesis python sample_vlm_reasoning_questions.py

# 2. Label  (keep terminal open while labeling in browser)
conda run -n myo_thesis python labeling_app.py
#    → open http://localhost:5050

# 3. Stats  (run any time, even mid-session)
conda run -n myo_thesis python compute_stats.py
```

---

## Notes

- **Keep the app running** while labeling in the browser. Stop with `Ctrl+C`.
- Re-running Step 1 with a **different seed** changes the sampled set — redo
  labeling accordingly if you change seeds.
- `compute_stats.py` can be re-run at any time for partial results.
- All three scripts resolve paths relative to their own location, so they
  work correctly regardless of the working directory.
