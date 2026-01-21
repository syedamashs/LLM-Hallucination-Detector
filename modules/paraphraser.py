import requests
import re

def paraphrase_llm(question):
    try:
        url = "http://localhost:11434/api/generate"

        prompt = f"""
Give me exactly 3 paraphrased versions of this question.
Write each in a new line without numbering.

Question: {question}
"""

        data = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=data, timeout=20)
        result = response.json()["response"]

        lines = result.strip().split("\n")
        paraphrases = [re.sub(r'^\d+[\.\)]\s*', '', line.strip()) 
               for line in lines if len(line.strip()) > 5]

        if len(paraphrases) >= 3:
            return paraphrases[:3]
        else:
            raise Exception("Not enough paraphrases")

    except Exception as e:
        print("LLM failed, using fallback...")
        return paraphrase_fallback(question)


def paraphrase_fallback(question):
    return [
        question,
        "Can you tell me " + question.lower(),
        "What is the answer to: " + question
    ]


def paraphrase(question):
    return paraphrase_llm(question)


# TEST
if __name__ == "__main__":
    print("Enter the Phrase to ParaPhrase:")
    q = input()
    paras = paraphrase(q)
    for i, p in enumerate(paras, 1):
        print(f"{i}. {p}")
