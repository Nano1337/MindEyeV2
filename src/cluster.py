import numpy as np
from cuml.cluster.dbscan import DBSCAN # ,  all_points_membership_vectors
from cuml.cluster import AgglomerativeClustering
from cuml.manifold import UMAP
import torch

import pandas as pd

# path = "clip_image_embeddings_10k.npy"
path = "/home/haoli/Documents/vlm-clustering/embeddings_10k.npy"
embeddings = np.load(path)


if "1k" in path: 
    embeddings = embeddings.squeeze(1)
    embeddings = np.mean(embeddings, axis=1)

# l2 normalize embeddings 
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# # reduce dim to 5 with UMAP
print("reducing dim with UMAP")
umap = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=12, metric='cosine')
reduced_data = umap.fit_transform(embeddings)

# # cluster using Agglomerative Clustering
# print("clustering with Agglomerative Clustering")
# # You can adjust n_clusters, linkage, and affinity as needed
# clusterer = AgglomerativeClustering(n_clusters=10, linkage='single', affinity='cosine')
# clusterer.fit(reduced_data)
# pd.Series(clusterer.labels_).value_counts()

# cluster using DBSCAN
print("clustering with DBSCAN")
clusterer = DBSCAN(eps=0.5, min_samples=20, metric='cosine')
clusterer.fit(reduced_data)
pd.Series(clusterer.labels_).value_counts()

# Create a DataFrame with indices and cluster labels
results_df = pd.DataFrame({'index': range(len(clusterer.labels_)), 'cluster_label': clusterer.labels_})

# Save the DataFrame to a CSV file
print("saving results to csv")
results_df.to_csv('clustering_results_10k_again.csv', index=False)

