import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from paraphraser import paraphrase
from llm_client import get_llm_answer
from embedding import get_embeddings
from similarity import compute_hcs, detect_hallucination
from entropy_baseline import entropy_baseline   # entropy baseline
from cross_model_baseline import cross_model_baseline  # mistral vs llama3

# SBERT for evaluation
eval_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_true_label(llm_answers, correct_answers, threshold=0.75):
    """
    Semantic ground truth using SBERT.
    If at least 2 LLM answers match correct answers -> Not hallucination
    """
    correct_list = [c.strip() for c in correct_answers.split(";")]

    llm_emb = eval_model.encode(llm_answers)
    correct_emb = eval_model.encode(correct_list)

    sim_matrix = cosine_similarity(llm_emb, correct_emb)

    match_count = 0
    for i in range(len(llm_answers)):
        if np.max(sim_matrix[i]) >= threshold:
            match_count += 1

    if match_count >= 2:
        return 0  # Not hallucination
    else:
        return 1  # Hallucination


def evaluate_system(csv_path, n_samples=10):
    df = pd.read_csv(csv_path)
    df = df.sample(n=n_samples, random_state=42)  # RANDOM 50

    true_labels = []
    preds_entropy = []
    preds_cross = []
    preds_hcs = []

    for i, (_, row) in enumerate(df.iterrows()):
        question = row["Question"]
        correct_answers = row["Correct Answers"]

        print("\n=================================================")
        print(f"Question {i+1}: {question}")

        # ---------- HCS METHOD ----------
        paraphrases = paraphrase(question)
        answers = [get_llm_answer(p) for p in paraphrases]
        embeddings = get_embeddings(answers)
        hcs = compute_hcs(embeddings)
        pred_hcs = detect_hallucination(hcs)
        pred_hcs = 1 if pred_hcs else 0

        print("\nHCS Answers:")
        for j, a in enumerate(answers, 1):
            print(f"Answer {j}: {a}")
        print("HCS Score:", round(hcs, 3))
        print("HCS Prediction:", "Hallucination" if pred_hcs else "Not Hallucination")

        # ---------- ENTROPY BASELINE ----------
        ans_entropy = get_llm_answer(question)
        pred_entropy = entropy_baseline(ans_entropy)

        print("\nEntropy Answer:")
        print(ans_entropy)
        print("Entropy Prediction:", "Hallucination" if pred_entropy else "Not Hallucination")

        # ---------- CROSS-MODEL BASELINE ----------
        a1, a2, sim, pred_cross = cross_model_baseline(question)

        print("\nCross-Model Answers:")
        print("Mistral:", a1)
        print("Llama3:", a2)
        print("Cross Similarity:", round(sim, 3))
        print("Cross Prediction:", "Hallucination" if pred_cross else "Not Hallucination")

        # ---------- TRUE LABEL ----------
        true = get_true_label(answers, correct_answers)
        print("\nTrue Label:", "Hallucination" if true else "Not Hallucination")

        true_labels.append(true)
        preds_entropy.append(pred_entropy)
        preds_cross.append(pred_cross)
        preds_hcs.append(pred_hcs)

    # ---------- METRICS ----------
    def print_metrics(name, preds):
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)

        print(f"\n==== {name} ====")
        print("Accuracy :", round(acc, 3))
        print("Precision:", round(prec, 3))
        print("Recall   :", round(rec, 3))
        print("F1 Score :", round(f1, 3))

    print("\n\n================ FINAL COMPARISON ================")
    print_metrics("Entropy Baseline", preds_entropy)
    print_metrics("Cross-Model Baseline", preds_cross)
    print_metrics("HCS (Our Method)", preds_hcs)


if __name__ == "__main__":
    evaluate_system("../TruthfulQA.csv", n_samples=50)
