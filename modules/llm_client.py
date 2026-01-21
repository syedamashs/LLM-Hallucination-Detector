import requests

def get_llm_answer(question):
    url = "http://localhost:11434/api/generate"

    prompt = f"""
Answer the following question clearly and concisely:

{question}
"""

    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data, timeout=20)
    result = response.json()["response"]

    return result.strip()


def get_multiple_answers(questions):
    answers = []
    for q in questions:
        ans = get_llm_answer(q)
        answers.append(ans)
    return answers


# TEST
if __name__ == "__main__":
    qs = [
        "Who is the CEO of Google in 2035?",
        "Who will lead Google in 2035?",
        "Name the Google CEO in 2035."
    ]

    answers = get_multiple_answers(qs)
    for i, a in enumerate(answers, 1):
        print(f"\nAnswer {i}:\n{a}")
