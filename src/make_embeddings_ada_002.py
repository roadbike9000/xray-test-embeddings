# make_embeddings_ada_002.py
# src\make_embeddings_ada_002.py
from openai import OpenAI
import re
import os
import numpy as np
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


# Function to get embeddings from OpenAI
def get_openai_embeddings(texts):
    """
    Generates embeddings for a list of texts using OpenAI's text-embedding-ada-002 model.

    Args:
        texts (list of str): A list of text strings for which embeddings are to be generated.

    Returns:
        list of list of float: A list where each element is a list of floats representing the embedding of the corresponding input text.
    """
    embeddings = []
    for text in texts:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
    return embeddings


# Generate embeddings
print("Generate embeddings using OpenAI")
embeddings_ada = get_openai_embeddings(test_cases)

# Save the embeddings as a numpy file
print("Save the embeddings as a numpy file")
np.save('embeddings_ada.npy', embeddings_ada)
