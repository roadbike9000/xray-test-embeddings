# Cluster using DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# Function to apply DBSCAN with cosine distance
def dbscan_clustering(embeddings, eps=0.5, min_samples=5):
    cosine_sim = cosine_similarity(embeddings)
    distance = 1 - cosine_sim
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance)
    return labels


# Cluster the embeddings from OpenAI
labels_ada = dbscan_clustering(embeddings_ada)

# Cluster the embeddings from Sentence-Transformer
labels_sentence_transformer = dbscan_clustering(embeddings_sentence_transformer)

# Now compare the cluster labels (labels_ada and labels_sentence_transformer)
