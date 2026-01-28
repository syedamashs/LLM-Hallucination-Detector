from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_hcs(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    
    n = sim_matrix.shape[0]
    total = 0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            total += sim_matrix[i][j]
            count += 1
    
    hcs = total / count
    return hcs


def detect_hallucination(hcs, threshold=0.75):
    if hcs < threshold:
        return True   # Hallucination
    else:
        return False  # Not hallucination


# TEST
if __name__ == "__main__":
    import numpy as np
    fake_embeddings = np.random.rand(3, 384)
    print("HCS:", compute_hcs(fake_embeddings))
