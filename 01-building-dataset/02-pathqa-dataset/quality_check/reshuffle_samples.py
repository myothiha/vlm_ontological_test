import json
import random
import os

def resample_and_shuffle(file_path, subset_size=50, seed=42):
    """
    Reads the sampled_concepts.json file, extracts all available concepts,
    takes a completely unbiased random sample, and then shuffles the order.
    The remaining elements are placed into a 'disabled_concepts' list.
    """
    
    with open(file_path, "r") as f:
        data = json.load(f)

    # Collect all concepts from both active and disabled arrays
    all_concepts = data.get("concepts", []) + data.get("disabled_concepts", [])
    
    # Sort them first to ensure deterministic selection when applying the seed
    all_concepts.sort()

    # Set random seed to ensure reproducibility
    random.seed(seed)
    
    # Take a fair, unbiased sample
    sampled_subset = random.sample(all_concepts, min(subset_size, len(all_concepts)))
    
    # Any concept not picked goes to disabled
    disabled_subset = [c for c in all_concepts if c not in sampled_subset]

    # Shuffle the resulting arrays so they don't default back to alphabetical
    random.shuffle(sampled_subset)
    random.shuffle(disabled_subset)

    # Reconstruct the dictionary
    new_data = {
        "seed": seed,
        "n": len(sampled_subset),
        "concepts": sampled_subset,
        "disabled_concepts": disabled_subset
    }

    # Write safely back to the JSON file
    with open(file_path, "w") as f:
        json.dump(new_data, f, indent=2)
        
    print(f"Successfully resampled and reshuffled: {file_path}")
    print(f"   --> Active Concepts: {len(sampled_subset)}")
    print(f"   --> Disabled Concepts: {len(disabled_subset)}")

if __name__ == "__main__":
    # Ensure it works dynamically based on where the script lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_json = os.path.join(current_dir, "data", "sampled_concepts.json")
    
    if os.path.exists(target_json):
        # By default we are resampling 50 back out using seed 42
        resample_and_shuffle(target_json, subset_size=50, seed=42)
    else:
        print(f"Error: Could not find the file at {target_json}")
