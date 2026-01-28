from llm_client import get_llm_answer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def cross_model_baseline(question):
    ans1 = get_llm_answer(question, model="mistral")
    ans2 = get_llm_answer(question, model="llama3")

    emb = model.encode([ans1, ans2])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

    if sim < 0.75:
        return ans1, ans2, sim, 1  # Hallucination
    else:
        return ans1, ans2, sim, 0  # Not Hallucination


# ---- RUN ----
if __name__ == "__main__":
    q = input("Enter your question: ")

    a1, a2, sim, pred = cross_model_baseline(q)

    print("\nMistral Answer:\n", a1)
    print("\nLlama3 Answer:\n", a2)
    print("\nSimilarity:", round(sim, 3))

    if pred:
        print("Cross-Model Result: Hallucination ❌")
    else:
        print("Cross-Model Result: Not Hallucination ✅")
