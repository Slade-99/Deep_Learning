import pandas as pd

def remove_rows_with_or_in_labels(csv_file_path, output_file_path):
    """
    Reads a CSV file, removes rows where the 'labels' column contains '|', and saves the modified DataFrame to a new CSV.

    Args:
        csv_file_path (str): The path to the input CSV file.
        output_file_path (str): The path to save the output CSV file.
    """
    try:
        df = pd.read_csv(csv_file_path)

        # Check if 'labels' column exists
        if 'View Position' not in df.columns:
            print(f"Error: 'labels' column not found in {csv_file_path}")
            return

        # Create a boolean mask to identify rows where 'labels' does not contain '|'
        mask = ~df['View Position'].str.contains('AP', regex=False, na=False) # regex=False to treat | literally

        # Apply the mask to filter the DataFrame
        filtered_df = df[mask]

        # Save the filtered DataFrame to a new CSV file
        filtered_df.to_csv(output_file_path, index=False)

        print(f"Rows with 'AP' in 'View Position' removed. Saved to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse CSV file at {csv_file_path}. Please ensure it is a valid CSV.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
input_csv = '/mnt/hdd/dataset_collections/Data_Entry_2017_2.csv'  # Replace with your input CSV file path
output_csv = '/mnt/hdd/dataset_collections/Data_Entry_2017_3.csv' # Replace with desired output file path
remove_rows_with_or_in_labels(input_csv, output_csv)