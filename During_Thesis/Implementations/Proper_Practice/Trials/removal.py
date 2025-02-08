import pandas as pd

# Load the CSV file into a DataFrame
file_path = '/home/azwad/Works/PadChest/PadChest_dataset_full/modified.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Replace values in the third column (index 2) where cell contains ['normal'] with 'normal'
df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: 'normal' if str(x) == "['normal']" else x)

# Replace any cell in the third column containing 'pneumonia' with 'pneumonia'
df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: 'pneumonia' if 'pneumonia' in str(x).lower() else x)

# Replace any cell in the third column that is not 'normal' or 'pneumonia' with 'abnormal'
df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: 'abnormal' if x not in ['normal', 'pneumonia'] else x)

# Save the modified DataFrame back to a new CSV file
df.to_csv('/home/azwad/Works/PadChest/PadChest_dataset_full/modified2.csv', index=False)  # Save to a new file (or overwrite if preferred)
