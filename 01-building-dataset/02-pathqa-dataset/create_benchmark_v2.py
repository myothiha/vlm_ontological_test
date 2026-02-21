# create_benchmark_v2.py
# Updated benchmark creation script that utilizes the knowledge_base folder structure
# instead of the single monolithic 03_generate_vlm_reasoning_questions.csv file.
#
# Knowledge base structure:
#   results/knowledge_base/
#     concept_vlm_reasoning_questions.csv   (index: concept -> relative paths per dimension)
#     <concept_folder>/
#       vlm_reasoning_questions/
#         properties.csv
#         functions.csv
#         relational.csv
#         contexual_properties.csv
#         contexual_behavior.csv

import torch
import os
import sys

from PIL import Image
import imagehash

# Add the parent folder to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gc
import json
import random
import pandas as pd

from dotenv import load_dotenv
from collections import defaultdict

from PIL import Image

from datasets import load_dataset, Dataset, Features, Value, Image
from huggingface_hub import login
import shutil

from libs.concept_extractor.medical_concept_extractor import MedicalConceptExtractor
from libs.dataset_loader.processed_pathvqa_dataset_loader import PathVQADatasetLoader
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper
from libs.llm_loader.llm_wrapper.gpt_llm_wrapper import GPTLLMWrapper
from libs.concept_extractor.llm_based_medical_concept_extractor import LLMBasedMedicalConceptExtractor
from libs.concept_extractor.medical_concept_classifier import MedicalConceptClassifier
from libs.OntologyBenchmarkBuilder.pipeline import OntoBenchPipeline
from libs.utils import setup_logger

# ──────────────────────────────────────────────────────────────────────────────
# KnowledgeBaseLoader: reads the nested knowledge_base folder structure
# ──────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_DIMENSIONS = [
    "properties",
    "functions",
    "relational",
    "contexual_properties",
    "contexual_behavior",
]


class KnowledgeBaseLoader:
    """
    Loads VLM reasoning questions from the knowledge_base folder structure.

    Provides the same conceptual API as the old VLMReasoningQuestionsLoader:
      - get_positive_questions(concept)  -> dict[dimension] -> list[str]
      - get_negative_questions(concept, exclude_concepts, num_concepts, num_questions)
                                          -> dict[dimension] -> list[str]
    """

    def __init__(self, knowledge_base_dir, concept_bench=None, logger=None):
        """
        Args:
            knowledge_base_dir: Path to the knowledge_base root folder.
            concept_bench: OntoBenchPipeline instance for generating questions
                           for concepts that are not yet in the knowledge base.
            logger: Logger instance for logging messages.
        """
        self.kb_dir = os.path.abspath(knowledge_base_dir)
        self.pipeline = concept_bench
        self.logger = logger

        # Build the index of available concepts from the index CSV
        self.concept_index = self._build_concept_index()

    # ── Index building ────────────────────────────────────────────────────

    def _build_concept_index(self):
        """
        Read the concept_vlm_reasoning_questions.csv index file to discover
        which concepts are available and where their dimension CSVs live.

        Returns a dict:  concept_name -> {dimension_name -> absolute_csv_path}
        """
        index_csv = os.path.join(self.kb_dir, "concept_vlm_reasoning_questions.csv")
        if not os.path.exists(index_csv):
            if self.logger:
                self.logger.warning(f"Index CSV not found: {index_csv}. Starting with empty knowledge base.")
            return {}

        df = pd.read_csv(index_csv)
        # Drop duplicate rows (the index CSVs in this project sometimes have duplicates)
        df = df.drop_duplicates(subset=["concept"])

        concept_index = {}
        for _, row in df.iterrows():
            concept = row["concept"]
            dim_paths = {}
            for dim in KNOWLEDGE_DIMENSIONS:
                if dim in row and pd.notna(row[dim]):
                    dim_paths[dim] = os.path.join(self.kb_dir, row[dim])
            concept_index[concept] = dim_paths

        if self.logger:
            self.logger.info(f"KnowledgeBaseLoader: loaded index for {len(concept_index)} concepts.")
        return concept_index

    # ── Reading dimension CSVs ────────────────────────────────────────────

    def _read_dimension_questions(self, csv_path):
        """
        Read a single dimension CSV file and return a list of question strings.
        Each dimension CSV has a single column called 'content'.
        """
        if not os.path.exists(csv_path):
            return []
        try:
            df = pd.read_csv(csv_path)
            if "content" in df.columns:
                return df["content"].dropna().tolist()
            else:
                # Fallback: use the first column
                return df.iloc[:, 0].dropna().tolist()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reading {csv_path}: {e}")
            return []

    def _load_concept_questions(self, concept):
        """
        Load all VLM reasoning questions for a concept, organized by dimension.

        Returns: dict[dimension_name] -> list[str]
        """
        if concept not in self.concept_index:
            return {}

        result = {}
        for dim, csv_path in self.concept_index[concept].items():
            result[dim] = self._read_dimension_questions(csv_path)
        return result

    # ── Public API ────────────────────────────────────────────────────────

    def get_available_concepts(self):
        """Return a list of all concept names in the knowledge base."""
        return list(self.concept_index.keys())

    def has_concept(self, concept):
        """Check if a concept exists in the knowledge base."""
        return concept in self.concept_index

    def _has_questions(self, concept):
        """Check if a concept has at least one non-empty dimension."""
        if concept not in self.concept_index:
            return False
        for dim, csv_path in self.concept_index[concept].items():
            questions = self._read_dimension_questions(csv_path)
            if len(questions) > 0:
                return True
        return False

    def get_positive_questions(self, concept):
        """
        Get positive VLM reasoning questions for a concept (questions whose
        expected answer is "Yes" because the concept is present in the image).

        If the concept is unknown but a pipeline is available, generate on-the-fly
        and save into the knowledge base folder structure.

        Returns: dict[dimension_name] -> list[str]
        """
        if concept not in self.concept_index:
            if self.logger:
                self.logger.info(f"Concept '{concept}' not in knowledge base. "
                                 f"{'Generating via pipeline...' if self.pipeline else 'Skipping.'}")
            if self.pipeline:
                self._generate_and_save_concept(concept)
            else:
                return {dim: [] for dim in KNOWLEDGE_DIMENSIONS}

        return self._load_concept_questions(concept)

    def get_negative_questions(self, concept, exclude_concepts=None, num_concepts=3, num_questions=10):
        """
        Get negative VLM reasoning questions: sample reasoning questions from
        other concepts (not related to the current image) whose expected
        answer is "No".

        Args:
            concept: The current concept to exclude.
            exclude_concepts: Additional concepts to exclude (e.g. all concepts
                              associated with the current image).
            num_concepts: Number of other concepts to sample.
            num_questions: Max number of questions to sample per dimension per concept.

        Returns: dict[dimension_name] -> list[str]
        """
        available = self._get_negative_concept_pool(concept, exclude_concepts)

        if len(available) < num_concepts:
            if self.logger:
                self.logger.warning(
                    f"Only {len(available)} negative concepts available (need {num_concepts}). "
                    f"Using all available."
                )
            sampled_concepts = available
        else:
            sampled_concepts = random.sample(available, num_concepts)

        if self.logger:
            self.logger.info(f"Negative concepts for '{concept}': {sampled_concepts}")

        # Collect questions across sampled concepts, grouped by dimension
        negative_questions = {dim: [] for dim in KNOWLEDGE_DIMENSIONS}
        for neg_concept in sampled_concepts:
            concept_qs = self._load_concept_questions(neg_concept)
            for dim in KNOWLEDGE_DIMENSIONS:
                dim_qs = concept_qs.get(dim, [])
                sample_size = min(num_questions, len(dim_qs))
                if sample_size > 0:
                    negative_questions[dim].extend(random.sample(dim_qs, sample_size))

        return negative_questions

    def _get_negative_concept_pool(self, concept, exclude_concepts=None):
        """
        Return a list of concepts suitable for negative sampling:
        excludes the target concept, any explicitly excluded concepts, and
        any concepts with no questions.
        """
        exclude_set = set(exclude_concepts or [])
        if concept is not None:
            exclude_set.add(concept)
        return [
            c for c in self.concept_index.keys()
            if c not in exclude_set and self._has_questions(c)
        ]

    # ── On-the-fly generation (fallback) ──────────────────────────────────

    def _generate_and_save_concept(self, concept):
        """
        Use the OntoBenchPipeline to generate conceptual knowledge for an
        unknown concept and save the VLM reasoning question CSVs
        into the knowledge_base folder structure.
        """
        if self.logger:
            self.logger.info(f"Generating knowledge for concept: {concept}")

        try:
            result = self.pipeline(concept)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Pipeline generation failed for '{concept}': {e}")
            return

        # Parse the generated VLM reasoning questions
        vlm_rq_raw = result.get("vlm_reasoning_questions", "{}")
        if isinstance(vlm_rq_raw, str):
            try:
                vlm_rq = json.loads(vlm_rq_raw)
            except json.JSONDecodeError:
                vlm_rq = {}
        else:
            vlm_rq = vlm_rq_raw

        # Create concept folder
        concept_folder = concept.lower().replace(" ", "_")
        concept_dir = os.path.join(self.kb_dir, concept_folder, "vlm_reasoning_questions")
        os.makedirs(concept_dir, exist_ok=True)

        # Save each dimension CSV
        dim_paths = {}
        for dim in KNOWLEDGE_DIMENSIONS:
            questions = vlm_rq.get(dim, [])
            csv_path = os.path.join(concept_dir, f"{dim}.csv")
            pd.DataFrame({"content": questions}).to_csv(csv_path, index=False)
            # Store the relative path (relative to knowledge_base root)
            dim_paths[dim] = os.path.join(concept_folder, "vlm_reasoning_questions", f"{dim}.csv")

        # Update the index CSV
        self.concept_index[concept_folder] = {
            dim: os.path.join(self.kb_dir, rel_path)
            for dim, rel_path in dim_paths.items()
        }

        # Append to the master index CSV
        index_csv = os.path.join(self.kb_dir, "concept_vlm_reasoning_questions.csv")
        new_row = {"concept": concept_folder}
        new_row.update(dim_paths)
        pd.DataFrame([new_row]).to_csv(index_csv, mode="a", header=False, index=False)

        if self.logger:
            self.logger.info(f"Saved generated knowledge for '{concept}' -> {concept_folder}/")


# ──────────────────────────────────────────────────────────────────────────────
# Main script
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Setup logger
    logger = setup_logger("CreateBenchmarkV2", os.path.join("results", "create_benchmark_v2.log"))

    # Load PathVQA dataset
    vqa_loader = PathVQADatasetLoader(split="train")

    # Concept extractor setup
    medical_concept_classifier = MedicalConceptClassifier(OllamaWrapper(model="gpt-oss:20b"))
    concept_extractor = LLMBasedMedicalConceptExtractor(
        model=OllamaWrapper(model="gpt-oss:20b"),
        backup_extractor=MedicalConceptExtractor("en_core_sci_scibert"),
        concept_classifier=medical_concept_classifier,
    )

    # Setup pipeline for generating questions for unknown concepts
    llm = GPTLLMWrapper("gpt-4.1")

    concept_bench_pipeline = OntoBenchPipeline(
        llm=llm,
        knowledge_question_prompt_templates="prompt_templates/kq_prompt_templates",
        generate_knowledge_prompt_template="prompt_templates/ck_prompt_templates",
        vlm_reasoning_prompt_template="prompt_templates/rq_prompt_templates",
    )

    # ── Load from knowledge_base folder instead of single CSV ──
    knowledge_base_dir = "results/knowledge_base"
    kb_loader = KnowledgeBaseLoader(
        knowledge_base_dir=knowledge_base_dir,
        concept_bench=concept_bench_pipeline,
        logger=logger,
    )

    logger.info(f"Knowledge base loaded with {len(kb_loader.get_available_concepts())} concepts.")

    # ── Build the benchmark ──────────────────────────────────────────────
    new_data = []
    for row in vqa_loader.sample(n=1):
        img_hash = imagehash.phash(row["image"])
        questions_and_answers = row["questions_and_answers"]
        text = row["text"]

        all_concepts = set()

        for question_answer in questions_and_answers:
            question = question_answer["question"]
            answer = question_answer["answer"]

            # Extract concepts from the question and answer
            combined_text = f"{question} {answer}"
            current_concepts = concept_extractor.extract(combined_text)
            current_concepts = list(set(current_concepts))  # Remove duplicates

            question_answer["extracted_concepts"] = current_concepts
            all_concepts.update(current_concepts)

        all_concepts = list(set(all_concepts))

        logger.info(f"All concepts for image {img_hash}: {all_concepts}")

        multiconcept_reasoning_questions = dict()

        for concept in all_concepts:
            try:
                exclude_concepts = [c for c in all_concepts if c != concept]

                # Get positive questions (concept IS in the image → answer = Yes)
                positive_questions = kb_loader.get_positive_questions(concept)

                # Get negative questions (concept NOT in the image → answer = No)
                negative_questions = kb_loader.get_negative_questions(
                    concept,
                    exclude_concepts=exclude_concepts,
                    num_concepts=10,
                    num_questions=15,
                )

                vlm_reasoning_questions = {
                    "yes_questions": positive_questions,
                    "no_questions": negative_questions,
                }

                # Log counts per dimension
                yes_total = sum(len(qs) for qs in positive_questions.values())
                no_total = sum(len(qs) for qs in negative_questions.values())
                logger.info(f"Concept: {concept}")
                logger.info(f"  Yes questions (total): {yes_total}")
                for dim, qs in positive_questions.items():
                    logger.info(f"    {dim}: {len(qs)}")
                logger.info(f"  No questions (total): {no_total}")
                for dim, qs in negative_questions.items():
                    logger.info(f"    {dim}: {len(qs)}")

                multiconcept_reasoning_questions[concept] = vlm_reasoning_questions

            except Exception as e:
                logger.error(f"Questions Generation Error for '{concept}': {e}")
                continue

        new_row = dict()
        new_row["image"] = row["image"]
        new_row["questions_and_answers"] = questions_and_answers
        new_row["all_concepts"] = json.dumps(all_concepts)
        new_row["multiconcept_reasoning_questions"] = json.dumps(multiconcept_reasoning_questions)
        logger.info(f"Built row for image hash={img_hash} with {len(all_concepts)} concepts")
        new_data.append(new_row)

    logger.info(f"Total benchmark rows: {len(new_data)}")

    # Create new dataset
    new_dataset = Dataset.from_list(new_data)

    # STEP: Authenticate and push to Hub
    login(token=os.getenv("HF_ACCESS_TOKEN"))

    try:
        new_dataset.push_to_hub("myothiha/conceptbench_path_vqa")
        logger.info("✅ Dataset pushed successfully to Hugging Face.")
    except Exception as e:
        logger.error(f"❌ Failed to push dataset: {e}")
        logger.warning("⚠️ Keeping local files for debugging.")
