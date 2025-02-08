from sklearn.model_selection import KFold
import time
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
from preprocessing import train_data
from Model.model import model
import torch.nn.functional as F  
from torch import optim
from utils import save_checkpoint,load_checkpoint
current_datetime = datetime.now()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 3
learning_rate = 0.0001
batch_size = 16
num_epochs = 5
num_rounds = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)


### Training Loop  ##
log_file_path = "/home/azwad/Works/Deep_Learning/Implementations/architecture_weights/runs_description.txt"
description = "Implementation of CusTomV11 architecture with 3.9M weights"
name = "CustomV11"




# Train Network
def kfold_cross_validation(k, dataset, batch_size, model):
    # Split the dataset into k folds
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []
    accuracy = 0
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Training Fold {fold+1}/{k}...")

        # Create the train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize your model here (e.g., a simple CNN or pre-trained model)
        model = model  # Replace with your model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        
        for epoch in range(num_epochs):  
            model.train()
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            time.sleep(12)


        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        results.append(accuracy)
        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

    print(f"Average Accuracy: {np.mean(results):.4f}")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    with open(log_file_path, 'a') as f:
      f.write("\n\n\n")
      f.write(f"Training completed at {current_datetime}\n")
      f.write("=================================\n")
      f.write(f"{description}\n")
      f.write(f"Accuracy obtained = {accuracy}\n")
    save_checkpoint(checkpoint,name)
    return results


results = kfold_cross_validation(num_rounds, train_data, batch_size,model)
