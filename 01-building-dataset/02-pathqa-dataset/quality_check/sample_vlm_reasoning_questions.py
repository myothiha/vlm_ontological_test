"""
Quality Check: Sample VLM Reasoning Questions
==============================================
Randomly selects a specified number of concepts (default: 50) that have
non-empty vlm_reasoning_questions, then writes their questions to
paginated output text files (10 concepts per file).

Usage:
    python sample_vlm_reasoning_questions.py [--n 50] [--seed 42]

Output:
    quality_check/outputs/sample_<i>.txt   (10 concepts per file)
"""

import os
import json
import random
import argparse
import csv
from pathlib import Path
from typing import Dict, List

# ─── Configuration ────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = Path(__file__).parent.parent / "results" / "knowledge_base"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# The dimension CSV files inside each concept's vlm_reasoning_questions folder
DIMENSION_FILES = [
    "properties.csv",
    "functions.csv",
    "relational.csv",
    "contexual_properties.csv",
    "contexual_behavior.csv",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_questions_from_csv(csv_path: Path) -> List[str]:
    """Return all non-empty question strings from a CSV with a 'content' column."""
    questions = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("content", "").strip()
                if q:
                    questions.append(q)
    except Exception as e:
        print(f"  [WARN] Could not read {csv_path}: {e}")
    return questions


def load_concept_questions(concept_dir: Path) -> Dict[str, List[str]]:
    """
    Load all dimension questions for a concept.
    Returns a dict: {dimension_name: [question, ...]}
    Only includes dimensions that have at least one question.
    """
    vlm_dir = concept_dir / "vlm_reasoning_questions"
    if not vlm_dir.is_dir():
        return {}

    dimensions = {}
    for fname in DIMENSION_FILES:
        fpath = vlm_dir / fname
        if fpath.exists():
            qs = read_questions_from_csv(fpath)
            if qs:
                dim_name = fpath.stem.replace("_", " ").title()
                dimensions[dim_name] = qs
    return dimensions


def concept_has_questions(concept_dir: Path) -> bool:
    """Quick check: does this concept have at least one non-empty VLM question?"""
    vlm_dir = concept_dir / "vlm_reasoning_questions"
    if not vlm_dir.is_dir():
        return False
    for fname in DIMENSION_FILES:
        fpath = vlm_dir / fname
        if fpath.exists() and read_questions_from_csv(fpath):
            return True
    return False


# ─── Core Logic ───────────────────────────────────────────────────────────────

def get_valid_concepts(kb_path: Path) -> list[Path]:
    """Return all concept directories that have at least one VLM reasoning question."""
    valid = []
    for entry in sorted(kb_path.iterdir()):
        if entry.is_dir() and concept_has_questions(entry):
            valid.append(entry)
    return valid


def format_concept_block(concept_name: str, dimensions: Dict[str, List[str]]) -> str:
    """Format one concept's questions as a readable text block."""
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"  CONCEPT: {concept_name.upper().replace('_', ' ')}")
    lines.append(sep)

    for dim_name, questions in dimensions.items():
        lines.append(f"\n  [{dim_name}]")
        for i, q in enumerate(questions, 1):
            lines.append(f"    {i}. {q}")

    lines.append("")
    return "\n".join(lines)


def write_output_files(
    sampled: list[Path],
    output_dir: Path,
    per_file: int = 10,
) -> None:
    """Write sampled concepts to paginated output text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 1
    chunk_start = 0

    while chunk_start < len(sampled):
        chunk = sampled[chunk_start : chunk_start + per_file]
        out_path = output_dir / f"sample_{file_idx:02d}.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"VLM Reasoning Questions – Quality Check Sample\n")
            f.write(f"File {file_idx} | Concepts {chunk_start + 1}–{chunk_start + len(chunk)}\n")
            f.write("=" * 70 + "\n\n")

            for concept_dir in chunk:
                dimensions = load_concept_questions(concept_dir)
                block = format_concept_block(concept_dir.name, dimensions)
                f.write(block + "\n")

        print(f"  Written: {out_path}")
        file_idx += 1
        chunk_start += per_file


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample VLM reasoning questions from the knowledge base for quality checking."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of concepts to randomly sample (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--per-file",
        type=int,
        default=10,
        help="Number of concepts per output file (default: 10).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing sampled concepts and generate a completely fresh sample.",
    )
    args = parser.parse_args()

    print(f"Knowledge base: {KNOWLEDGE_BASE}")
    print(f"Scanning for concepts with non-empty VLM reasoning questions...")

    valid_concepts = get_valid_concepts(KNOWLEDGE_BASE)
    print(f"Found {len(valid_concepts)} valid concepts (out of total dirs in KB).")

    if args.n > len(valid_concepts):
        print(
            f"[WARN] Requested {args.n} concepts but only {len(valid_concepts)} are valid. "
            f"Sampling all of them."
        )
        args.n = len(valid_concepts)

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    manifest_path = data_dir / "sampled_concepts.json"

    # 1. Load any concepts we've already sampled to avoid losing work
    existing_names = set()
    if manifest_path.exists() and not args.reset:
        try:
            old_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_names = set(old_data.get("concepts", []))
        except Exception:
            pass

    # 2. Partition valid concepts into (Already Sampled) and (Unsampled Pool)
    already_sampled_paths = [c for c in valid_concepts if c.name in existing_names]
    pool = [c for c in valid_concepts if c.name not in existing_names]

    # 3. Determine how many new concepts we need
    sampled = already_sampled_paths[:args.n]  # truncate if args.n < existing
    needed = args.n - len(sampled)

    if needed > 0:
        effective_needed = min(needed, len(pool))
        if needed > len(pool):
            print(f"[WARN] Need {needed} more concepts but only {len(pool)} available.")
        
        random.seed(args.seed)
        new_sample = random.sample(pool, effective_needed)
        sampled.extend(new_sample)
        print(f"\nKept {len(already_sampled_paths)} existing concepts, and randomly added {effective_needed} new ones.")
    else:
        print(f"\nKept {len(sampled)} existing concepts (no new concepts needed).")

    sampled.sort(key=lambda p: p.name)  # sort alphabetically for readability
    
    print(f"\nTotal sample size is {len(sampled)} concepts:")
    for c in sampled:
        tag = "[NEW]" if c.name not in existing_names else "[EXISTING]"
        print(f"  - {c.name} {tag}")

    print(f"\nWriting output files to: {OUTPUT_DIR}")
    write_output_files(sampled, OUTPUT_DIR, per_file=args.per_file)

    # ── Save manifest for the labeling app ───────────────────────────────────
    manifest = {
        "seed": args.seed,
        "n": len(sampled),
        "concepts": [c.name for c in sampled],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved to: {manifest_path}")

    print(f"\nDone! {len(sampled)} concepts written across "
          f"{-(-len(sampled) // args.per_file)} file(s).")


if __name__ == "__main__":
    main()
