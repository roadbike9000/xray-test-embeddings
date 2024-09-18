from openai import OpenAI
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# Set the OpenAI API key for Axion
print("Set the OpenAI API key for Axion")
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)

# Load the file with explicit encoding
print("Load the file with explicit encoding")
file_path = r'C:\Users\Smath\Source\xray-test-embeddings\test_cases_output.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Split the content into test cases using the pattern 'NAV-' followed by digits
print("Split the content into test cases")
test_cases = re.split(r'(?=NAV-\d+)', content)

# Remove any empty strings from the list
print("Remove any empty strings from the list")
test_cases = [case.strip() for case in test_cases if case.strip()]

# Load the pre-trained model
print("Load the pre-trained model")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings using Sentence-Transformer
print("Generate embeddings using Sentence-Transformer")
embeddings_sentence_transformer = model.encode(test_cases)

# Save the embeddings as a numpy file
print("Save the embeddings as a numpy file")
np.save('embeddings_sentence_transformer.npy', embeddings_sentence_transformer)
