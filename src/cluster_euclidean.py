import numpy as np
from cuml.cluster.hdbscan import HDBSCAN, all_points_membership_vectors
from cuml.manifold import UMAP
import torch

import pandas as pd

path = "clip_image_embeddings_10k.npy"
embeddings = np.load(path)

if "1k" in path: 
    embeddings = embeddings.squeeze(1)
    embeddings = np.mean(embeddings, axis=1)

# # reduce dim to 5 with UMAP
print("reducing dim with UMAP")
umap = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=12, metric='euclidean')
reduced_data = umap.fit_transform(embeddings)

# cluster using HDBSCAN with Euclidean distance
print("clustering with HDBSCAN using Euclidean distance")
clusterer = HDBSCAN(min_cluster_size=50, metric='euclidean', prediction_data=True)
clusterer.fit(reduced_data)
soft_clusters = all_points_membership_vectors(clusterer)
pd.Series(clusterer.labels_).value_counts()

# Create a DataFrame with indices and cluster labels
results_df = pd.DataFrame({'index': range(len(clusterer.labels_)), 'cluster_label': clusterer.labels_})

# Save the DataFrame to a CSV file
print("saving results to csv")
results_df.to_csv('clustering_results_10k_euclidean.csv', index=False)

