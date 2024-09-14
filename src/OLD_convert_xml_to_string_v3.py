# Description: Convert XML data to string format
# Convert the string to a list
# Calculate the embeddings for each test case string, then
# Cluster the embeddings to group similar test cases together
# Version: 3.0 save strings to file

import xml.etree.ElementTree as ET
import re
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# Set the OpenAI API key for Axion
openai.api_key = os.environ['OPENAI_API_KEY']

# Parse the XML file
tree = ET.parse('master_test_cases.xml')
root = tree.getroot()
total_test_cases = len(root.findall('Test'))
total_tests = 0


# Function to clean the action text
def clean_action_text(text):
    return re.sub(r'!xray-attachment:.*?!', '', text)


# Open a file to write the test case strings
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
        total_tests += 1

print(f"Total test cases to process: {total_test_cases}")
print(f"Total test cases processed: {total_tests}")

# Load test case strings from file
with open('test_cases_output.txt', 'r', encoding='utf-8') as file:
    test_case_strings = file.readlines()

# Create a list removing any leading or trailing whitespaces
test_case_strings = [test_case.strip() for test_case in test_case_strings]


# Calculate the embeddings for each test case string
# Function to generate embeddings
def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-3-small"
    )
    return response['data'][0]['embedding']


# Calculate embeddings for all test cases then print embeddings
embeddings = [get_embeddings(test_case) for test_case in test_case_strings]
print(embeddings)
