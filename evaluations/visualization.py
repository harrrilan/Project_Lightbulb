import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import csv

def plot_embeddings_3d_tsne(embeddings_path, chunks_path):
    # Load embeddings and chunk texts
    embeddings = np.load(embeddings_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    #kmeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)


    # Gaussian Mixture Model clustering
    # so far set with 5
    #gmm = GaussianMixture(n_components=4, random_state=42)
    #cluster_labels = gmm.fit_predict(embeddings)


    # Save cluster labels to CSV
    output_data = [
        {"index": i, "text": chunk, "cluster": int(cluster_labels[i])}
        for i, chunk in enumerate(chunks)
    ]
    with open("chunk_clusters.csv", "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["index", "text", "cluster"])
        writer.writeheader()
        writer.writerows(output_data)

    # Apply t-SNE to reduce to 3D
    #tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    #embeddings_3d = tsne.fit_transform(embeddings)  # 3D
    
    # Apply UMAP to reduce to 3D
    # Uncomment this block to use UMAP instead of t-SNE
    #reducer = umap.UMAP(n_components=3, random_state=42)
    #embeddings_3d = reducer.fit_transform(embeddings)

    # Apply PCA to reduce to 3D
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings)

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        color=cluster_labels,
        hover_name=[f"Chunk {i}" for i in range(len(chunks))],
        hover_data={"Text": chunks},
        title="3D Visualization of BERT Embeddings clustered by GMM, reduced by PCA"
    )
    fig.show()


plot_embeddings_3d_tsne("all_embeddings.npy", "all_chunks.json")