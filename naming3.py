import os

# Define the folder path
folder = '/home/azwad/Downloads/Completed'

# Strings to check for
keywords = ['pneumonia', 'normal', 'abnormal', 'effusion']

# Get the list of files from the folder
files = os.listdir(folder)

# Process each file and print the ones that don't contain the keywords
for file in files:
    if not any(keyword in file.lower() for keyword in keywords):
        print(file)
