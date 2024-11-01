import matplotlib.pyplot as plt

# Data
labels = ['Normal', 'Abnormal', 'Pneumonia', 'Effusion']
values = [622, 345, 219, 14]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])

# Adding titles and labels
plt.title('Image Classification Distribution', fontsize=16)
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)

# Show the values on top of the bars
for i, value in enumerate(values):
    plt.text(i, value + 10, str(value), ha='center', fontsize=12)

# Show the chart
plt.tight_layout()
plt.show()
