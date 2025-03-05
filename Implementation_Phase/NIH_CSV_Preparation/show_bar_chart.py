import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(csv_path, label_column='Finding Labels', output_plot_path='class_distribution.png'):
    """
    Reads a CSV file, counts class occurrences, and generates a bar chart.

    Args:
        csv_path (str): The path to the CSV file.
        label_column (str): The name of the column with labels.
        output_plot_path (str): The path to save the generated plot.
    """

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    label_counts = df[label_column].value_counts().sort_values(ascending=False) #sort values for better visualization

    plt.figure(figsize=(12, 6)) #adjust figure size
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis") #use seaborn for better aesthetics.
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.tight_layout() #prevent labels from being cut off.

    try:
        plt.savefig(output_plot_path)
        print(f"Plot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show() #show plot in window

# Example Usage:
csv_file_path = '/mnt/hdd/dataset_collections/Data_Entry_2017_5.csv'  # Replace with your CSV file path
plot_class_distribution(csv_file_path) #uses default output path, or you can specify your own.