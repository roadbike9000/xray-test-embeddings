# compare_embeddings_models.py
# src\compare_embeddings_models.py
# Cluster using DBSCAN
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
import umap
import random
import string


'''
Run DBSCAN on the Generated Embeddings
Once you have the embeddings from both models (embeddings_ada and 
embeddings_sentence_transformer), you can apply DBSCAN to cluster them.
'''


# Now compare the cluster labels (labels_ada and labels_sentence_transformer
# Function to apply DBSCAN with cosine distance
def dbscan_clustering(embeddings, eps=0.4, min_samples=5):
    cosine_sim = cosine_similarity(embeddings)
    # Clip cosine similarity values to avoid numerical errors
    cosine_sim = np.clip(cosine_sim, 0, 1)
    distance = 1 - cosine_sim
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance)
    return labels


# Load the embeddings from the numpy file
embeddings_ada = np.load('embeddings_ada.npy')
# Cluster the embeddings from OpenAI
eps = 0.1
min_samples = 2
labels_ada = dbscan_clustering(embeddings_ada, eps, min_samples)

# Load the embeddings from the numpy file
# embeddings_sentence_transformer = np.load('embeddings_sentence_transformer.npy')
# Cluster the embeddings from Sentence-Transformer
# labels_sentence_transformer = dbscan_clustering(embeddings_sentence_transformer)


# Compare the cluster labels (labels_ada and labels_sentence_transformer)

# 1. Count how many clusters DBSCAN generated and how many points
# were labeled as noise (i.e., -1).


# Function to count clusters and noise points
def analyze_dbscan_labels(labels):
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise (-1)
    num_noise_points = list(labels).count(-1)
    return num_clusters, num_noise_points


# Analyze and print results for OpenAI embeddings
num_clusters_ada, num_noise_ada = analyze_dbscan_labels(labels_ada)
print(f"OpenAI Model: {num_clusters_ada} clusters, {num_noise_ada} noise points")

# Analyze and print results for Sentence-Transformer embeddings
# num_clusters_st, num_noise_st = analyze_dbscan_labels(labels_sentence_transformer)
# print(f"Sentence-Transformer Model: {num_clusters_st} clusters, {num_noise_st} noise points")


# 2. Calculate the cluster size distribution. (i.e., how many points are in each cluster).


# Function to calculate cluster sizes
def cluster_size_distribution(labels):
    cluster_sizes = Counter(labels)
    return cluster_sizes


# Analyze and print cluster sizes for OpenAI embeddings
cluster_sizes_ada = cluster_size_distribution(labels_ada)
print(f"OpenAI Cluster Sizes: {dict(cluster_sizes_ada)}")

# Analyze and print cluster sizes for Sentence-Transformer embeddings
# cluster_sizes_st = cluster_size_distribution(labels_sentence_transformer)
# print(f"Sentence-Transformer Cluster Sizes: {dict(cluster_sizes_st)}")

# 3. Intra-cluster Similarity
# Assess how well the clusters group similar items,
# calculate the average cosine similarity between all pairs
# of points in each cluster.
# Function to calculate intra-cluster similarity


def intra_cluster_similarity(embeddings, labels):
    unique_labels = set(labels)
    intra_cluster_sim = {}
    
    # Calculate cosine similarity
    cosine_sim_matrix = cosine_similarity(embeddings)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        # Get indices of points in this cluster
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        if len(indices) > 1:
            # Calculate average similarity within the cluster
            intra_sim = cosine_sim_matrix[np.ix_(indices, indices)].mean()
        else:
            intra_sim = 1.0  # If only one point, it's perfectly similar to itself
        intra_cluster_sim[label] = intra_sim
    
    return intra_cluster_sim


# Calc and print Intra-cluster similarity for OpenAI embeddings
intra_cluster_sim_ada = intra_cluster_similarity(embeddings_ada, labels_ada)
print(f"OpenAI Intra-cluster Similarity: {intra_cluster_sim_ada}")

# Calc and print Intra-cluster similarity for Sentence-Transformer embeddings
# intra_cluster_sim_st = intra_cluster_similarity(embeddings_sentence_transformer, labels_sentence_transformer)
# print(f"Sentence-Transformer Intra-cluster Similarity: {intra_cluster_sim_st}")

# 4. Measure Inter-cluster Distance
# Determine how distinct the clusters are from each other.
# Calculate the average distance (1 - cosine similarity) between cluster centroids.
# Function to calculate cluster centroids


def calculate_centroids(embeddings, labels):
    unique_labels = set(labels)
    centroids = {}

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        cluster_embeddings = np.array([embeddings[i] for i in indices])
        centroid = cluster_embeddings.mean(axis=0)  # Calculate mean vector (centroid)
        centroids[label] = centroid

    return centroids


# Function to calculate inter-cluster distances
def inter_cluster_distance(centroids):
    cluster_labels = list(centroids.keys())
    num_clusters = len(cluster_labels)
    distances = {}

    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            c1, c2 = centroids[cluster_labels[i]], centroids[cluster_labels[j]]
            sim = cosine_similarity([c1], [c2])[0][0]
            dist = 1 - sim  # Convert similarity to distance
            distances[(cluster_labels[i], cluster_labels[j])] = dist

    return distances

# Calculate and print inter-cluster distance for OpenAI embeddings
# centroids_ada = calculate_centroids(embeddings_ada, labels_ada)
# inter_cluster_dist_ada = inter_cluster_distance(centroids_ada)
# print(f"OpenAI Inter-cluster Distances: {inter_cluster_dist_ada}")
# Function to generate a random string
def generate_random_string(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Generate a random string and append it to the file name
random_string = generate_random_string()
file_name = f'inter_cluster_dist_ada_{random_string}.txt'

# Write the inter-cluster distances to the file with the random name
with open(file_name, 'w') as f:
    f.write(str(file_name))

# Calculate and print inter-cluster distance for Sentence-Transformer embeddings
# centroids_st = calculate_centroids(embeddings_sentence_transformer, labels_sentence_transformer)
# inter_cluster_dist_st = inter_cluster_distance(centroids_st)
# print(f"Sentence-Transformer Inter-cluster Distances: {inter_cluster_dist_st}")

# UMAP Visualization for ADA embeddings
# First, you’ll run UMAP to reduce the dimensionality of your data and
# visualize it in 2D space. This will give us insight into the structure
# of your data and guide the tuning of DBSCAN

# Assuming 'embeddings' is your high-dimensional data (from ADA or sentence-transformers)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)  # Use UMAP to reduce to 2D
reduced_embeddings = reducer.fit_transform(embeddings_ada)

# Example: Using cluster labels for coloring
# Assuming 'labels' is an array of cluster labels for the embeddings
labels = dbscan_clustering(embeddings_ada)


# Plot the UMAP visualization
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', s=5)
plt.title(f'UMAP Visualization of the ADA Embeddings {eps} {min_samples}')
plt.show()

# UMAP Visualization for ST embeddings
# First, you’ll run UMAP to reduce the dimensionality of your data and
# visualize it in 2D space. This will give us insight into the structure
# of your data and guide the tuning of DBSCAN

# # Assuming 'embeddings' is your high-dimensional data (from ADA or sentence-transformers)
# reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42)  # Use UMAP to reduce to 2D
# reduced_embeddings = reducer.fit_transform(embeddings_sentence_transformer)

# # Plot the UMAP visualization
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], cmap='Spectral', s=5)
# plt.title('UMAP Visualization of the ST Embeddings')
# plt.show()
