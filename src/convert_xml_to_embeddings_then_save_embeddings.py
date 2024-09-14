# Description: Convert XML data to string format
# Convert the string to a list
# Calculate the embeddings for each test case string, then
# Cluster the embeddings to group similar test cases together
# Version: 3.0 save strings to file

import xml.etree.ElementTree as ET
import re
import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# Set the OpenAI API key for Axion
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)


# Parse the XML file
tree = ET.parse('master_test_cases.xml')
root = tree.getroot()
total_test_cases = len(root.findall('Test'))
total_tests = 0


# Function to clean the action text
def clean_action_text(text):
    return re.sub(r'!xray-attachment:.*?!', '', text)


# Open a file to write the test case strings file
# file name is test_cases_output.txt
""" print("Writing test case strings to file")
with open('test_cases_output.txt', 'w', encoding='utf-8') as file:
    # Iterate over each test case
    for test_case in root.findall('Test'):
        # Extract relevant elements
        test_case_id = test_case.find('JiraKey').text
        jira_summary = test_case.find('JiraSummary').text
        steps = "; ".join(
            [
                clean_action_text(step.find('Action').text)
                for step in test_case.find('Steps').findall('TestStep')
            ]
        )

        # Construct the string
        test_case_string = f"{test_case_id}, {jira_summary}, {steps}"

        # Write the string to the file
        file.write(test_case_string + '\n')
        total_tests += 1 """

print(f"Total test cases to process: {total_test_cases}")
print(f"Total test cases processed: {total_tests}")

# Load test case strings from file
print("Loading test case strings from file")
with open('test_cases_output.txt', 'r', encoding='utf-8') as file:
    test_case_strings = file.readlines()

# Create a list removing any leading or trailing whitespaces
print("Creating a list of test case strings")
test_case_strings = [test_case.strip() for test_case in test_case_strings]


# Calculate the embeddings for each test case string
# Function to generate embeddings
def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


# Calculate embeddings for all test cases then print time taken
print("Calculating embeddings for all test cases")
start_time = time.time()  # Record start time
all_embeddings = [get_embeddings(test_case) for test_case in test_case_strings]
elapsed_time = time.time() - start_time  # Calculate elapsed time
print(f"Elapsed time to calculate embeddings: {elapsed_time / 60} minutes")

# Process test cases in batches
# save embeddings to a NumPy array
# save the array to a file
# print embeddings for each batch
# batch_size = 100  # Number of test cases to process in each batch
# all_embeddings = []

# for i in range(0, len(test_case_strings), batch_size):
#     start_time = time.time()  # Record start time
#     batch = test_case_strings[i:i + batch_size]
#     batch_embeddings = [get_embeddings(test_case) for test_case in batch]
#     # Calculate elapsed time for the batch
#     elapsed_time = time.time() - start_time
#     print(f"Batch {i//batch_size + 1} embeddings: {batch_embeddings}")
#     print(f"Elapsed time for batch {i//batch_size + 1}: {elapsed_time} seconds")

#     # Append the embeddings to the list
#     all_embeddings.extend(batch_embeddings)

# Convert the list of embeddings to a NumPy array
print("Converting the list of embeddings to a NumPy array")
all_embeddings = np.array(all_embeddings)

# Save the embeddings to a file
print("Saving the embeddings to a file")
np.save('test_case_embeddings.npy', all_embeddings)
