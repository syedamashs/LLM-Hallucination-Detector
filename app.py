from modules.paraphraser import paraphrase
from modules.llm_client import get_multiple_answers
from modules.embedding import get_embeddings
from modules.similarity import compute_hcs, detect_hallucination

def main():
    print("\n===== AI Hallucination Detection System =====\n")

    print("Enter your question:")
    question = input()

    # Step 1: Paraphrasing
    print("\n--- Step 1: Paraphrasing ---")
    paraphrases = paraphrase(question)
    for i, p in enumerate(paraphrases, 1):
        print(f"{i}. {p}")

    # Step 2: LLM Responses
    print("\n--- Step 2: LLM Responses ---")
    answers = get_multiple_answers(paraphrases)
    for i, a in enumerate(answers, 1):
        print(f"\nAnswer {i}:\n{a}")

    # Step 3: Embeddings
    print("\n--- Step 3: Embedding Generation ---")
    embeddings = get_embeddings(answers)
    print("Embeddings Shape:", embeddings.shape)

    # Step 4: Similarity + HCS
    print("\n--- Step 4: Similarity Computation ---")
    hcs = compute_hcs(embeddings)
    print("HCS Score:", round(hcs, 3))

    # Step 5: Final Decision
    print("\n--- Final Result ---")
    is_hallu = detect_hallucination(hcs)
    if is_hallu:
        print("Result: Hallucination Detected ❌")
    else:
        print("Result: No Hallucination ✅")

    print("\n============================================\n")

if __name__ == "__main__":
    main()
