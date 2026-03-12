# VLM Ontological Test / PathQA Dataset Evaluation

This repository contains the codebase and data for building the PathQA dataset and evaluating various Vision-Language Models (VLMs) on their biomedical conceptual knowledge and reasoning capabilities.

## Repository Structure

- `01-building-dataset/`: Contains scripts, prompt templates, and tools to construct and refine the dataset (e.g., extracting five-dimensional conceptual knowledge).
- `02-evaluate-models/`: Includes code to load VLMs, execute evaluations on the dataset, and log the results.
- `libs/`: Contains Git submodules and shared utility code used across the project (e.g., wrapper for LLM/VLM loading and NVIDIA GPU monitoring).

## Installation and Setup

### 1. Cloning the Repository

**Important:** This project uses Git submodules (e.g., `llm_wrapper` and `nvidia_usage_info`). A standard `git clone` will not download the contents of these submodules. 

To properly clone the repository along with all its submodules, run the following command:

```bash
git clone --recurse-submodules git@github.com:myothiha/vlm_ontological_test.git
```

If you have already cloned the repository without the `--recurse-submodules` flag, you can initialize and update the submodules by running:

```bash
cd <repository-directory>
git submodule update --init --recursive
```

### 2. Setting Up the Environment

This project utilizes `conda` for dependency management. To set up your local environment:

1. Create the conda environment using the provided environment file:
   ```bash
   conda env create -f environment.yml
   # or use conda_env.yml if applicable
   ```

2. Activate the newly created environment:
   ```bash
   conda activate myo_thesis
   ```

3. Ensure you have the proper `.env` file set up (you can copy `.env.example` to `.env` and fill in necessary API keys and paths).

## Updating Submodules Later

If the upstream submodules are updated and you need to pull their latest changes, run:

```bash
git submodule update --remote --recursive
```

## Running Evaluations

Navigate to the `02-evaluate-models/` directory where you can define the VLMs to test in `models_config.py` and run the evaluation scripts such as `02_evaluate_answer.py`.
