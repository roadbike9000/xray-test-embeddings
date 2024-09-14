import xml.etree.ElementTree as ET
import os

# Directory containing the XML files
xml_directory = r'C:\Users\jsmathers\Downloads\xray_tests'

# Create the root element for the master XML
root = ET.Element('TestCases')

# Iterate over each XML file in the directory
for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        # Parse the current XML file
        tree = ET.parse(os.path.join(xml_directory, filename))
        test_case = tree.getroot()
        
        # Append the test case to the root element of the master XML
        root.append(test_case)

# Create a new tree with the combined root element
master_tree = ET.ElementTree(root)

# Save the combined XML to a file
master_tree.write('master_test_cases.xml', encoding='utf-8', xml_declaration=True)

print("All test cases have been combined into 'master_test_cases.xml'")
