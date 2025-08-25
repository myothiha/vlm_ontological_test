# both actual_concepts and extracted_concepts must not have any duplicates.
def relax_evaluation(actual_concepts, extracted_concepts):

    # relax match
    # an extracted concepts will be considered positive if they are exact match or substring of one the actual concepts.
    true_positives = sum(
        1 for concept in extracted_concepts
        if any(concept == actual or concept in actual for actual in actual_concepts)
    )

    # False positives: extracted concepts that are not exact or substring matches to any actual concept
    false_positives = sum(
        1 for concept in extracted_concepts
        if not any(concept == actual or concept in actual for actual in actual_concepts)
    )
    # False negatives: actual concepts that are not exact or substring matches to any extracted concept
    false_negatives = sum(
        1 for actual in actual_concepts
        if not any(actual == concept or concept in actual for concept in extracted_concepts)
    )

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }