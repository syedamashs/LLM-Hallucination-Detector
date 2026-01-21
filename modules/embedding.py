from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings


# TEST
if __name__ == "__main__":
    sample = [
        "I like dogs",
        "I love puppies",
        "The sky is blue"
    ]
    emb = get_embeddings(sample)
    print(emb.shape)   # (3, 384)
