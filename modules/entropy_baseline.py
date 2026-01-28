def entropy_baseline(answer):
    uncertain_words = [
        "maybe",
        "might",
        "possibly",
        "not sure",
        "cannot",
        "unknown",
        "uncertain",
        "hard to say",
        "i don't know",
        "cannot predict",
        "no information",
        "not possible"
    ]

    answer = answer.lower()

    for w in uncertain_words:
        if w in answer:
            return 1  # Hallucination

    return 0  # Not Hallucination


# ---- RUN STANDALONE ----
if __name__ == "__main__":
    print("Enter your question:")
    q = input()

    ans, pred = entropy_baseline(q)
    print("\nLLM Answer:\n", ans)

    if pred == 1:
        print("\nEntropy Baseline Result: Hallucination ❌")
    else:
        print("\nEntropy Baseline Result: Not Hallucination ✅")
