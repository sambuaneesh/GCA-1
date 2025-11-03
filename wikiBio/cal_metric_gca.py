# sambuaneesh-gca-1/wikiBio/cal_metric_gca.py
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Configuration ---
# This is the output file from your score_with_rgcn.py run
SCORED_DATA_PATH = "/home/stealthspectre/iiith/GCA/Extract triples/processed/scored_wiki.json"


def calculate_passage_scores(scored_data):
    """
    Calculates a single score for each passage by averaging its triple scores.
    The scores are logits, so we apply a sigmoid to get probabilities [0, 1].
    A higher score means more plausible/factual.
    """
    passage_scores = []
    labels = []
    for entry in scored_data:
        triple_scores = entry.get("triples_score", [])

        # Handle entries with no scores (e.g., empty graphs)
        if not triple_scores:
            # Assign a neutral or low score for entries with no triples to score
            passage_score = 0.5
        else:
            # Convert logits to probabilities using sigmoid
            probabilities = torch.sigmoid(torch.tensor(triple_scores)).numpy()
            # Average the probabilities
            passage_score = np.mean(probabilities)

        passage_scores.append(passage_score)
        labels.append(entry["label"])

    return np.array(passage_scores), labels


def evaluate_at_threshold(scores, true_labels, threshold):
    """
    Makes predictions based on a threshold and computes metrics.
    If score > threshold, it's predicted as non-factual (hallucination).
    """
    # The paper's goal is to detect hallucinations.
    # Higher score = more plausible = factual. Lower score = less plausible = non-factual.
    # Let's define the labels: factual=0, non-factual=1
    label_map = {"factual": 0, "non-factual": 1}

    # Prediction: 1 (non-factual) if score is BELOW threshold, 0 (factual) if above.
    # This aligns with the idea that low plausibility scores indicate hallucinations.
    predicted_labels_numeric = (scores < threshold).astype(int)

    true_labels_numeric = [label_map[label] for label in true_labels]

    f1 = f1_score(true_labels_numeric, predicted_labels_numeric, zero_division=0)
    acc = accuracy_score(true_labels_numeric, predicted_labels_numeric)
    precision = precision_score(true_labels_numeric, predicted_labels_numeric, zero_division=0)
    recall = recall_score(true_labels_numeric, predicted_labels_numeric, zero_division=0)

    return {
        "F1": f1,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
    }


def main():
    print(f"Loading scored data from {SCORED_DATA_PATH}...")
    try:
        with open(SCORED_DATA_PATH, "r", encoding="utf-8") as f:
            scored_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scored data file not found at {SCORED_DATA_PATH}")
        print("Please make sure you have run 'score_with_rgcn.py' successfully.")
        return

    # 1. Calculate passage scores for all entries
    passage_scores, true_labels = calculate_passage_scores(scored_data)

    # 2. Calculate mean (μ) and standard deviation (σ) of the scores
    mu = np.mean(passage_scores)
    sigma = np.std(passage_scores)
    print(f"\nScore statistics: Mean (μ) = {mu:.4f}, Std Dev (σ) = {sigma:.4f}")

    # 3. Sweep through different thresholds (μ + k·σ)
    print("--- Evaluating at different thresholds (Threshold = μ - k·σ) ---")

    best_f1 = -1
    best_metrics = {}
    best_k = None

    # Sweeping k from negative to positive values to find the best threshold
    # The paper seems to imply a sweep, let's test a reasonable range for k.
    # A negative k means the threshold will be > mu, classifying more things as hallucinations.
    # Let's try k from -2 to 2.
    for k in np.linspace(-2.0, 2.0, 9):  # e.g., -2.0, -1.5, ..., 1.5, 2.0
        # We use mu - k*sigma because a LOWER score should indicate hallucination.
        # A low score falling below a low threshold is the detection condition.
        threshold = mu - k * sigma

        metrics = evaluate_at_threshold(passage_scores, true_labels, threshold)

        print(f"\nFor k = {k:+.2f} (Threshold = {threshold:.4f}):")
        print(
            f"  F1: {metrics['F1']:.4f}, Accuracy: {metrics['Accuracy']:.4f}, "
            f"Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}"
        )

        # Keep track of the best result based on F1-score
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_metrics = metrics
            best_k = k

    print("\n" + "=" * 50)
    print("           Best Result (max F1-score)")
    print("-" * 50)
    print(f"Achieved at k = {best_k:+.2f} (Threshold = {mu - best_k * sigma:.4f})")
    print(f"  - F1-score: {best_metrics['F1']:.4f}")
    print(f"  - Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"  - Precision: {best_metrics['Precision']:.4f}")
    print(f"  - Recall:   {best_metrics['Recall']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
