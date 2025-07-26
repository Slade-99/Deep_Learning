import pandas as pd

def trim_and_filter_labels(csv_path, label_column='Finding Labels', min_count=1000, max_count=1500):
    """
    Trims high-frequency labels and filters low-frequency labels in a CSV file.

    Args:
        csv_path (str): The path to the CSV file.
        label_column (str): The name of the column with labels.
        min_count (int): Minimum samples per class.
        max_count (int): Maximum samples per class.

    Returns:
        pd.DataFrame: The trimmed and filtered DataFrame.
    """

    df = pd.read_csv(csv_path)

    label_counts = df[label_column].value_counts()

    # Filter Low-Frequency Labels
    valid_labels = label_counts[label_counts >= min_count].index.tolist()
    df = df[df[label_column].isin(valid_labels)]

    label_counts = df[label_column].value_counts() #recalculate counts after filtering.

    trimmed_dfs = []
    for label, count in label_counts.items():
        if count > max_count:
            temp_df = df[df[label_column] == label].sample(n=max_count, random_state=42)
            trimmed_dfs.append(temp_df)
        else:
            trimmed_dfs.append(df[df[label_column] == label])

    trimmed_df = pd.concat(trimmed_dfs)
    return trimmed_df

# Example Usage:
csv_file_path = '/mnt/hdd/dataset_collections/Data_Entry_2017_4.csv'  # Replace with your CSV file path
trimmed_df = trim_and_filter_labels(csv_file_path)

# Save the trimmed and filtered DataFrame to a new CSV file (optional):
trimmed_df.to_csv('/mnt/hdd/dataset_collections/Data_Entry_2017_5.csv', index=False) #replace with desired output path

print(trimmed_df.head()) #print the head of the trimmed and filtered data.