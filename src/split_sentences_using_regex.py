import re
from sentence_transformers import SentenceTransformer, util

# Load the file with explicit encoding
file_path = r'C:\Users\Smath\Source\xray-test-embeddings\test_cases_output.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Split the content into test cases using the pattern 'NAV-' followed by digits
test_cases = re.split(r'(?=NAV-\d+)', content)

# Remove any empty strings from the list
test_cases = [case.strip() for case in test_cases if case.strip()]

# Initialize the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Compute embeddings for all test cases
embeddings = model.encode(test_cases)

# Compute cosine similarities
similarities = util.pytorch_cos_sim(embeddings, embeddings)

# Output the pairs with their score
for idx_i, test_case1 in enumerate(test_cases):
    print(f"Test Case {idx_i + 1}: {test_case1.strip()}")
    for idx_j, test_case2 in enumerate(test_cases):
        if idx_i != idx_j:  # Skip self-similarity
            print(f" - Test Case {idx_j + 1}: {test_case2.strip()} : {similarities[idx_i][idx_j]:.4f}")