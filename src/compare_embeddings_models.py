# compare_embeddings_models.py
# src\compare_embeddings_models.py
# Cluster using DBSCAN
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap


'''
Run DBSCAN on the Generated Embeddings
Once you have the embeddings from both models (embeddings_ada and 
embeddings_sentence_transformer), you can apply DBSCAN to cluster them.
'''


# Now compare the cluster labels (labels_ada and labels_sentence_transformer
# Function to apply DBSCAN with cosine distance
def dbscan_clustering(embeddings, eps=0.5, min_samples=5):
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
labels_ada = dbscan_clustering(embeddings_ada)

# Load the embeddings from the numpy file
embeddings_sentence_transformer = np.load('embeddings_sentence_transformer.npy')
# Cluster the embeddings from Sentence-Transformer
labels_sentence_transformer = dbscan_clustering(embeddings_sentence_transformer)


# Compare the cluster labels (labels_ada and labels_sentence_transformer)

# 1. Count how many clusters DBSCAN generated and how many points
# were labeled as noise (i.e., -1).


# Function to count clusters and noise points
def analyze_dbscan_labels(labels):
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise (-1)
    num_noise_points = list(labels).count(-1)
    return num_clusters, num_noise_points


# Analyzing results for OpenAI embeddings
num_clusters_ada, num_noise_ada = analyze_dbscan_labels(labels_ada)
print(f"OpenAI Model: {num_clusters_ada} clusters, {num_noise_ada} noise points")

# Analyzing results for Sentence-Transformer embeddings
num_clusters_st, num_noise_st = analyze_dbscan_labels(labels_sentence_transformer)
print(f"Sentence-Transformer Model: {num_clusters_st} clusters, {num_noise_st} noise points")


# 2. Calculate the cluster size distribution. (i.e., how many points are in each cluster).


# Function to calculate cluster sizes
def cluster_size_distribution(labels):
    cluster_sizes = Counter(labels)
    return cluster_sizes


# Analyze cluster sizes for OpenAI embeddings
cluster_sizes_ada = cluster_size_distribution(labels_ada)
print(f"OpenAI Cluster Sizes: {dict(cluster_sizes_ada)}")

# Analyze cluster sizes for Sentence-Transformer embeddings
cluster_sizes_st = cluster_size_distribution(labels_sentence_transformer)
print(f"Sentence-Transformer Cluster Sizes: {dict(cluster_sizes_st)}")

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


# Intra-cluster similarity for OpenAI embeddings
intra_cluster_sim_ada = intra_cluster_similarity(embeddings_ada, labels_ada)
print(f"OpenAI Intra-cluster Similarity: {intra_cluster_sim_ada}")

# Intra-cluster similarity for Sentence-Transformer embeddings
intra_cluster_sim_st = intra_cluster_similarity(embeddings_sentence_transformer, labels_sentence_transformer)
print(f"Sentence-Transformer Intra-cluster Similarity: {intra_cluster_sim_st}")

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


# Calculate inter-cluster distance for OpenAI embeddings
centroids_ada = calculate_centroids(embeddings_ada, labels_ada)
inter_cluster_dist_ada = inter_cluster_distance(centroids_ada)
print(f"OpenAI Inter-cluster Distances: {inter_cluster_dist_ada}")

# Calculate inter-cluster distance for Sentence-Transformer embeddings
centroids_st = calculate_centroids(embeddings_sentence_transformer, labels_sentence_transformer)
inter_cluster_dist_st = inter_cluster_distance(centroids_st)
print(f"Sentence-Transformer Inter-cluster Distances: {inter_cluster_dist_st}")


'''
5. Compare the Results
The key metrics for comparison between the two models:

Number of clusters: Check how many clusters each model generated.
Noise points: See how many points were labeled as noise by each model.
Cluster sizes: Compare the distribution of cluster sizes for each model.
Intra-cluster similarity: Analyze how similar the points in each cluster are.
Inter-cluster distance: Compare how distinct the clusters are from each other.
'''

# TODO: Interpret the results
'''
Part 1: Interpreting the Results
You can compare the clustering performance of the models using the following
criteria:

Number of Clusters: A larger or smaller number of clusters isn’t necessarily
better—it depends on how naturally the test cases are grouped. If one model
has fewer clusters, it might be over-grouping; if it has many clusters, it
might be splitting them too finely.

Noise Points: Noise points are test cases that don't fit well into any cluster.
The model with fewer noise points might be better at identifying coherent
clusters, but too few noise points might suggest that the model is forcing
some points into clusters where they don’t belong.

Intra-cluster Similarity: Higher intra-cluster similarity means that the
points within each cluster are more alike. Look for higher average
intra-cluster similarity across clusters as a sign of a better model.

Inter-cluster Distance: Higher inter-cluster distances indicate that clusters
are more distinct from each other. The model that has larger distances between
clusters may better capture the semantic differences between test cases.

Key Questions for Interpretation:
Cluster Purity: Which model has higher intra-cluster similarity, meaning the
clusters contain more semantically similar test cases?

Cluster Separation: Which model has higher inter-cluster distances, meaning
the clusters are more distinct?
'''
# Visualize the clusters
# Dimensionality reduction to 2D for visualization using UMAP or TSNE


def reduce_dimensions(embeddings, method="umap"):
    if method == "umap":
        # original n_neighbors=15, min_dist=0.1
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    else:  # Use t-SNE if preferred
        reducer = TSNE(n_components=2, random_state=42)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


# Function to plot the clusters in 2D space
def plot_clusters(embeddings_2d, labels, title="Cluster Visualization"):
    # Define colors for clusters and noise points
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels) - 1)  # For clusters
    palette.append((0, 0, 0))  # Black color for noise points (-1)

    plt.figure(figsize=(10, 7))
    for cluster_label in unique_labels:
        if cluster_label == -1:  # Noise points
            color = "black"
            label = "Noise"
        else:
            color = palette[cluster_label]
            label = f"Cluster {cluster_label}"

        cluster_points = embeddings_2d[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=label, alpha=0.6, s=50)

    plt.title(title)
    plt.legend()
    plt.show()


# Reduce dimensions for OpenAI embeddings
reduced_ada = reduce_dimensions(embeddings_ada, method="umap")
plot_clusters(reduced_ada, labels_ada, title="Clusters from OpenAI Model (text-embedding-ada-002)")

# Reduce dimensions for Sentence-Transformer embeddings
# reduced_st = reduce_dimensions(embeddings_sentence_transformer, method="umap")
# plot_clusters(reduced_st, labels_sentence_transformer, title="Clusters from Sentence-Transformer Model (all-MiniLM-L6-v2)")
