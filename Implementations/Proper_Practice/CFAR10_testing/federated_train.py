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
import random
current_datetime = datetime.now()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 3
learning_rate = 0.0001
batch_size = 16
num_epochs = 5
num_rounds = 5
num_clients = 5
num_epochs_per_client = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)


### Training Loop  ##
log_file_path = "/home/azwad/Works/Deep_Learning/Implementations/architecture_weights/runs_description.txt"
description = "Implementation of CusTomV11 architecture with 3.9M weights"
name = "CustomV11"





### Federated Learning Setup ###
clients_data = [[] for _ in range(num_clients)]

def distribute_data(train_dataloader_pretrained):
    dataset = train_dataloader_pretrained.dataset
    total_samples = len(dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)  # Shuffle indices to randomize client data distribution

    data_per_client = total_samples // num_clients
    for client_idx in range(num_clients):
        start_idx = client_idx * data_per_client
        end_idx = (client_idx + 1) * data_per_client if client_idx != num_clients - 1 else total_samples
        clients_data[client_idx] = Subset(dataset, indices[start_idx:end_idx])

distribute_data(train_data)


accuracy_score = 0


# Train Network
####  Federated Learning Training Loop ####
for round_idx in range(num_rounds):
    print(f"Round {round_idx + 1}/{num_rounds}")
    selected_clients = random.sample(range(num_clients), num_clients // 2)
    client_models = []
    avg_accuracy = 0
    avg_loss = 0
    for client_idx in selected_clients:
        print(f"Client {client_idx + 1}/{num_clients} training...")

        local_model = model
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
        running_loss = 0.0
        all_preds = []
        all_labels = []

        client_dataloader = torch.utils.data.DataLoader(clients_data[client_idx], batch_size=32, shuffle=True)

        for epoch in range(num_epochs_per_client):
            print(f"Client {client_idx + 1} - Epoch {epoch + 1}/{num_epochs_per_client}")
            for images, labels in tqdm(client_dataloader, desc=f"Training Client {client_idx + 1}", unit="batch"):
                
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = local_model(images)
                
                loss = loss(outputs, labels)
                
                loss.backward()
                print("here")
                optimizer.step()
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss += running_loss / len(client_dataloader)
        avg_accuracy += accuracy_score(all_labels, all_preds)
        client_models.append(local_model.state_dict())

    avg_accuracy /= len(selected_clients)
    avg_loss /= len(selected_clients)

    global_model_state_dict = model.state_dict()
    for key in global_model_state_dict.keys():
        global_model_state_dict[key] = torch.mean(torch.stack([client_model[key].float() for client_model in client_models]), dim=0)

    model.load_state_dict(global_model_state_dict)



    # Log round details
    with open(log_file_path, 'a') as f:
        f.write(f"Round {round_idx + 1}:\n")
        f.write(f"  - Loss: {avg_loss:.4f}\n")
        f.write(f"  - Accuracy: {avg_accuracy:.4f}\n")