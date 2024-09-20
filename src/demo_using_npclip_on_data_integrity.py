import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# Example embeddings
embeddings = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0]
])

# Compute cosine similarity
cosine_sim = cosine_similarity(embeddings)

# Print original cosine similarity matrix
print("Original Cosine Similarity Matrix:")
print(cosine_sim)

# Clip cosine similarity values to the range [0, 1]
cosine_sim_clipped = np.clip(cosine_sim, 0, 1)

# Print clipped cosine similarity matrix
print("Clipped Cosine Similarity Matrix:")
print(cosine_sim_clipped)

# Convert to distance matrix
distance = 1 - cosine_sim_clipped

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
labels = dbscan.fit_predict(distance)

# Print cluster labels
print("Cluster Labels:")
print(labels)