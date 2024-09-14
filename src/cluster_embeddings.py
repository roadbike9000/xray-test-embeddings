from sklearn.cluster import DBSCAN
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


# Set the OpenAI API key for Axion
print("Set the OpenAI API key for Axion")
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)


# Load the embeddings from the file as a numpy array
print("Load the embeddings from the file as a numpy array")
loaded_embeddings = np.load('test_case_embeddings.npy')

dbscan = DBSCAN(
    eps=0.1,
    min_samples=5,
    metric='cosine'
)

print("Clustering the embeddings")
clustering = dbscan.fit(loaded_embeddings)

# Get the cluster labels
print("Get the cluster labels")
labels = clustering.labels_

# Calculate the number of clusters (excluding noise points labeled as -1)
print("Calculate the number of clusters")
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Print the number of clusters
print(f"Number of clusters identified: {num_clusters}")

# Load test case strings from file
print("Loading test case strings from file")
with open('test_cases_output.txt', 'r', encoding='utf-8') as file:
    test_case_strings = file.readlines()

# Create a list removing any leading or trailing whitespaces
print("Creating a list of test case strings")
test_case_strings = [test_case.strip() for test_case in test_case_strings]


# Print out the clusters
print("Printing out the clusters")
for i, label in enumerate(labels):
    print(f"Test Case {i} (Cluster {label}): {test_case_strings[i]}")
