# Description: Convert XML data to string format
# Version: 1.0 save strings to file
import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse('master_test_cases.xml')
root = tree.getroot()
total_test_cases = len(root.findall('Test'))
total_tests = 0
# Iterate over each test case
for test_case in root.findall('Test'):
    # Extract relevant elements
    test_case_id = test_case.find('JiraKey').text
    jira_summary = test_case.find('JiraSummary').text
    steps = "; ".join([step.find('Action').text for step in test_case.find('Steps').findall('TestStep')])

    # Construct the string
    test_case_string = f"{test_case_id}, {jira_summary}, {steps}"

    # Do something with the string (e.g., store, process, generate embeddings)
    print(test_case_string)
    total_tests += 1
print(f"Total test cases to process: {total_test_cases}")
print(f"Total test cases processed: {total_tests}")
