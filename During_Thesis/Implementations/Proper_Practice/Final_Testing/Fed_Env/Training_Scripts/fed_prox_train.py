from sklearn.model_selection import KFold
import time
from torch.utils.data import DataLoader, Subset
import torch
import copy
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
from Implementations.Preprocessings.Private_Dataset_Preprocessings.Fed_Train_prepare_private_dataset import  client_dataloaders
from Implementations.Preprocessings.Private_Dataset_Preprocessings.K_Fold_prepare_private_dataset_v2 import AugmentedDataset
from Implementations.Proper_Practice.Final_Testing.Model.Custom_Architecture.sparse_att import model
#from Implementations.Proper_Practice.Final_Testing.Model.Swin.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.LeViT.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.CVT.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.MobileViT_S.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.MobileNet_V2.model import model
import torch.nn.functional as F  
from torch import optim
from Implementations.Proper_Practice.Final_Testing.Utils.utils import save_checkpoint,load_checkpoint
import random
import matplotlib.pyplot as plt
import copy
current_datetime = datetime.now()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 3
learning_rate = 0.0001
batch_size = 16
fraction_clients = 0.5
num_rounds = 10
num_clients = 10
num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)


### Training Loop  ##

name = "Fed_Prox_Custom_Architecture"



global_model = model.to(device=device)
global_accuracies = []
global_losses = []


def evaluate_global_model(global_model, dataloaders):
    global_model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for dataloader in dataloaders:
            augmented_train_dataset = AugmentedDataset(dataloader.dataset,'val')
            dataloader = DataLoader(augmented_train_dataset, batch_size=16, shuffle=True)
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
    global_accuracy = 100 * total_correct / total_samples
    global_loss = total_loss / total_samples
    return global_accuracy, global_loss



def train_local_model(local_model, dataloader, num_epochs, criterion, optimizer, mu):
    local_model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0.0
        augmented_train_dataset = AugmentedDataset(dataloader.dataset,'aug')
        dataloader = DataLoader(augmented_train_dataset, batch_size=16, shuffle=True)
        for images, labels in tqdm(dataloader, desc=f"Client Training Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = local_model(images)
            loss = criterion(outputs, labels)

            # Proximal term: regularization between local and global parameters
            prox_term = torch.tensor(0.0, device=device)
            
            for (name, param), global_param in zip(local_model.named_parameters(), global_model.parameters()):
                # Ensure global parameters are detached and on the same device
                prox_term += ((param - global_param.detach()) ** 2).sum()
            
            # Add the proximal term
            loss += (mu / 2) * prox_term

            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2f}%")
    return local_model.state_dict()



def federated_average(client_weights, client_sizes):
    total_size = sum(client_sizes)
    averaged_weights = copy.deepcopy(client_weights[0])  # Deep copy to avoid modifying the original

    for key in averaged_weights.keys():
        # Initialize as float to avoid type mismatch
        averaged_weights[key] = torch.zeros_like(client_weights[0][key], dtype=torch.float32)

        # Weighted sum of model parameters
        for i in range(len(client_weights)):
            weight = client_weights[i][key].float() * (client_sizes[i] / total_size)
            averaged_weights[key] += weight

    return averaged_weights




for round_num in range(num_rounds):
    print(f"Round {round_num + 1}/{num_rounds}")

    # Select a subset of clients
    selected_clients = random.sample(range(num_clients), int(num_clients * fraction_clients))
    print(f"Selected clients: {selected_clients}")

    client_weights = []
    client_sizes = []

    # Train each selected client
    for client_id in selected_clients:
        print(f"Training client {client_id}")

        # Create a local copy of the global model
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(device=device)
        
        # Optimizer
        optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

        # Train the local model
        local_weights = train_local_model(local_model, client_dataloaders[client_id], num_epochs, criterion, optimizer , 0.001)
        client_weights.append(local_weights)
        client_sizes.append(len(client_dataloaders[client_id].dataset))

    # Federated averaging to update the global model
    
    global_weights = federated_average(client_weights,client_sizes)
    global_model.load_state_dict(global_weights)
    time.sleep(35)

    print(f"Completed round {round_num + 1}")

    global_accuracy, global_loss = evaluate_global_model(global_model, client_dataloaders)
    global_accuracies.append(global_accuracy)
    global_losses.append(global_loss)
    print(f"Global Accuracy: {global_accuracy:.2f}%, Global Loss: {global_loss:.4f}")
    
    
# Save the global model
checkpoint = {"state_dict": global_model.state_dict(), "optimizer": optimizer.state_dict()}
save_checkpoint(checkpoint,name)
print(f"Global model saved ")



















# Plot global accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_rounds + 1), global_accuracies, marker='o', label='Global Accuracy')
plt.title('Global Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.show()

# Plot global loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_rounds + 1), global_losses, marker='o', label='Global Loss')
plt.title('Global Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()