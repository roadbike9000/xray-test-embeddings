from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=1, metric='cosine')
clusters = dbscan.fit_predict(X)

print(clusters)

# Print sentences with their cluster labels
for sentence, cluster in zip(sentences, clusters):
    print(f"Cluster {cluster}: {sentence}")
