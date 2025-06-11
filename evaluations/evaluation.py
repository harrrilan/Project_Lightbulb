import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.load("all_embeddings.npy")

#===================================Quantitative Evaluation ===============================================

def get_top_n_similar(vector, embeddings, idx2word, top_n=5, exclude_indices=None):
    sims = cosine_similarity(vector.reshape(1, -1), embeddings).flatten()
    if exclude_indices:
        for i in exclude_indices:
            sims[i] = -np.inf  # Mask out inputs
    top_n_indices = sims.argsort()[-top_n:][::-1]
    return [(idx2word[i], sims[i]) for i in top_n_indices]

def evaluate_analogy_3cosadd(a_idx, b_idx, c_idx, embeddings, idx2word, top_n=5):
    """
    Given indices for A, B, C, find D such that A:B :: C:D using 3CosADD.
    Returns the top_n most similar candidates (excluding A, B, C).
    """
    # Compute the target vector: vec(B) - vec(A) + vec(C)
    target_vec = embeddings[b_idx] - embeddings[a_idx] + embeddings[c_idx]
    exclude_indices = {a_idx, b_idx, c_idx}
    # Use your existing get_top_n_similar function
    return get_top_n_similar(target_vec, embeddings, idx2word, top_n=top_n, exclude_indices=exclude_indices)

def evaluate_analogy_3cosmul(a_idx, b_idx, c_idx, embeddings, idx2word, top_n=5):
    """
    Given indices for A, B, C, find D such that A:B :: C:D using 3CosMUL.
    Returns the top_n most similar candidates (excluding A, B, C).
    """
    a_vec = embeddings[a_idx]
    b_vec = embeddings[b_idx]
    c_vec = embeddings[c_idx]
    exclude_indices = {a_idx, b_idx, c_idx}
    # Compute cosine similarities
    sim_b = cosine_similarity(b_vec.reshape(1, -1), embeddings).flatten()
    sim_c = cosine_similarity(c_vec.reshape(1, -1), embeddings).flatten()
    sim_a = cosine_similarity(a_vec.reshape(1, -1), embeddings).flatten()
    # 3CosMUL formula
    scores = (sim_b * sim_c) / (sim_a + 1e-8)
    if exclude_indices:
        for i in exclude_indices:
            scores[i] = -np.inf
    top_n_indices = scores.argsort()[-top_n:][::-1]
    return [(idx2word[i], scores[i], i) for i in top_n_indices]

# Example usage:
if __name__ == "__main__":
    import json
    with open("all_chunks.json", "r") as f:
        idx2word = json.load(f)

    # Define your test questions as (A_idx, B_idx, C_idx)
    test_questions = [
        (0, 1, 2),  # Example: chunk 0 is to chunk 1 as chunk 2 is to ?
        (3, 4, 5),  # Add your own indices here
        # Add more as needed
    ]

    for i, (a, b, c) in enumerate(test_questions):
        print(f"Input:")
        print(f"A: {idx2word[a]} [{a}]")
        print(f"B: {idx2word[b]} [{b}]")
        print(f"C: {idx2word[c]} [{c}]")
        print()
        print("================================================")
        # 3CosADD
        results_add = evaluate_analogy_3cosadd(a, b, c, embeddings, idx2word, top_n=3)
        print("output for Cos3add")
        for candidate, score in results_add:
            idx = idx2word.index(candidate)
            print(f"D{results_add.index((candidate, score))+1}: [{idx}]({candidate}) (score: {score:.4f})")
        print()
        print("================================================")

        # 3CosMUL
        results_mul = evaluate_analogy_3cosmul(a, b, c, embeddings, idx2word, top_n=3)
        print("output for Cos3mul")
        for candidate, score, idx in results_mul:
            print(f"D{results_mul.index((candidate, score, idx))+1}: [{idx}]({candidate}) (score: {score:.4f})")
        print()

#==========================================================================================================