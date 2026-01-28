import requests
import json
import os
from pathlib import Path

CACHE_FILE = "cache.json"


def get_cache_path():
    """Get the path to cache.json in the modules directory."""
    return Path(__file__).parent / CACHE_FILE


def load_cache():
    """Load cache from JSON file. Returns empty dict if file doesn't exist."""
    cache_path = get_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache):
    """Save cache to JSON file."""
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save cache: {e}")


def get_llm_answer(question, model="mistral"):
    """
    Get an LLM answer for a question.
    Checks cache first. If not found, calls Ollama API and caches result.
    
    Args:
        question: The question string
        model: The model name (default: "mistral")
    
    Returns:
        The answer from the LLM (cached or fresh)
    """
    # Load cache
    cache = load_cache()
    
    # Create cache key
    cache_key = f"{model}::{question}"
    
    # Check if answer is cached
    if cache_key in cache:
        return cache[cache_key]
    
    # Not cached - call Ollama API
    url = "http://localhost:11434/api/generate"

    prompt = f"""
Answer the following question clearly and concisely:

{question}
"""

    data = {
        "model": model,   # default is mistral
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data, timeout=120)
    result = response.json()["response"]
    answer = result.strip()
    
    # Cache the answer
    cache[cache_key] = answer
    save_cache(cache)
    
    return answer


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
