import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from modules.paraphraser import paraphrase
from modules.llm_client import get_multiple_answers
from modules.embedding import get_embeddings
from modules.similarity import compute_hcs, detect_hallucination


# Load SBERT once
eval_model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_match(llm_answers, correct_answers, threshold=0.7):
    """
    Returns True if at least 2 LLM answers
    are semantically similar to any correct answer.
    """
    correct_list = [c.strip() for c in correct_answers.split(";")]

    llm_emb = eval_model.encode(llm_answers)
    correct_emb = eval_model.encode(correct_list)

    sim_matrix = cosine_similarity(llm_emb, correct_emb)

    print("\nSimilarity with Correct Answers:")
    print(sim_matrix)

    match_count = 0
    for i in range(len(llm_answers)):
        if np.max(sim_matrix[i]) >= threshold:
            match_count += 1

    if match_count >= 2:
        return 0  # Not Hallucination (true label)
    else:
        return 1  # Hallucination (true label)


def evaluate_system(csv_path, n_samples=10):
    df = pd.read_csv(csv_path)

    true_labels = []
    predicted_labels = []

    for i in range(n_samples):
        row = df.iloc[i]
        question = row["Question"]
        correct_answers = row["Correct Answers"]

        print("\n=================================================")
        print(f"Question {i+1}: {question}")

        # Step 1: Paraphrase
        paraphrases = paraphrase(question)
        print("\nParaphrased Questions:")
        for j, p in enumerate(paraphrases, 1):
            print(f"{j}. {p}")

        # Step 2: LLM answers
        answers = get_multiple_answers(paraphrases)
        print("\nLLM Answers:")
        for j, a in enumerate(answers, 1):
            print(f"\nAnswer {j}: {a}")

        # Step 3: HCS (your system)
        embeddings = get_embeddings(answers)
        hcs = compute_hcs(embeddings)
        predicted = detect_hallucination(hcs)
        predicted = 1 if predicted else 0

        print("\nHCS Score:", round(hcs, 3))
        print("Predicted:", "Hallucination" if predicted else "Not Hallucination")

        # Step 4: True label (semantic evaluation)
        true = semantic_match(answers, correct_answers)

        print("True:", "Hallucination" if true else "Not Hallucination")

        true_labels.append(true)
        predicted_labels.append(predicted)

    # Final Metrics
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, zero_division=0)
    rec = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print("\n================ FINAL METRICS ================")
    print("Accuracy :", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall   :", round(rec, 3))
    print("F1 Score :", round(f1, 3))


# RUN
if __name__ == "__main__":
    evaluate_system("truthfulqa.csv", n_samples=1)
