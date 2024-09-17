from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# Sample data
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "Actions speak louder than words.",
    "Beauty is in the eye of the beholder.",
    "Better late than never.",
    "Birds of a feather flock together.",
    "A picture is worth a thousand words."
]

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=1, metric='cosine')
clusters = dbscan.fit_predict(embeddings)

print(clusters)

# Print sentences with their cluster labels
for sentence, cluster in zip(sentences, clusters):
    print(f"Cluster {cluster}: {sentence}")
