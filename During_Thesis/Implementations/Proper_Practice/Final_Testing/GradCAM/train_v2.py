from sklearn.model_selection import KFold
import time
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
import importlib
import During_Thesis.Implementations.Preprocessings.Private_Dataset_Preprocessings.K_Fold_prepare_private_dataset
from During_Thesis.Implementations.Preprocessings.Private_Dataset_Preprocessings.K_Fold_prepare_private_dataset_v2 import AugmentedDataset
#from Implementations.Proper_Practice.Final_Testing.Model.Custom_Architecture.sparse_att import model
#from Implementations.Proper_Practice.Final_Testing.Model.Swin.model import model
from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.LeViT.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.CVT.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.MobileViT_S.model_gradcam import model
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.MobileNet_V2.model_new import model
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.Swin.model_new import model
import torch.nn.functional as F  
from torch import optim
#from Implementations.Proper_Practice.Final_Testing.Utils.utils import save_checkpoint,load_checkpoint
current_datetime = datetime.now()


def save_checkpoint(state):
    filename="/home/azwad/Works/Model_Weights/LeViT"
    filename = filename+".pth.tar"
    print("=>Saving checkpoint")
    torch.save(state,filename)


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
log_file_path = "/home/azwad/Works/Deep_Learning/During_Thesis/Implementations/Proper_Practice/Final_Testing/GradCAM/runs_levit_new.txt"
description = "Implementation of LeViT on Benchmark dataset"
name = "LeViT on Benchmark"

"""
model_path = '/home/azwad/Works/Deep_Learning/Implementations/architecture_weights/Final Custom Architecture.pth.tar'
model = model.to(device)
learning_rate = 0.0001
checkpoint = torch.load(model_path)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(checkpoint['state_dict'])
"""

# Train Network
def kfold_cross_validation(k,  batch_size, model):
    
    importlib.reload(During_Thesis.Implementations.Preprocessings.Private_Dataset_Preprocessings.K_Fold_prepare_private_dataset)
    dataset = During_Thesis.Implementations.Preprocessings.Private_Dataset_Preprocessings.K_Fold_prepare_private_dataset.train_data

    
    
    # Split the dataset into k folds
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []
    accuracy_list = []
    loss_list = []
    epoch_list = []
    epoch_values = 1
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Training Fold {fold+1}/{k}...")

        # Create the train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        augmented_val_subset = AugmentedDataset(val_subset,'val')

        val_loader = DataLoader(augmented_val_subset, batch_size=16, shuffle=False)

        # Initialize your model here (e.g., a simple CNN or pre-trained model)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        
        
        for epoch in range(num_epochs):  
            model.train()
            count = 0
            augmented_train_subset = AugmentedDataset(train_subset, 'aug')
            train_loader = DataLoader(augmented_train_subset, batch_size=16, shuffle=True)
            
            correct = 0
            total = 0
            running_loss = 0.0  # To store cumulative loss
            
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                count += 1
                running_loss += loss.item()  # Accumulate loss
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                #if count % 50 == 0:
                #    time.sleep(20)
            
            epoch_loss = running_loss / len(train_loader)  # Compute average loss
            loss_list.append(epoch_loss)  # Store epoch-wise average loss

            epoch_list.append(epoch_values)
            epoch_values += 1
            
            #time.sleep(20)
            #loss_list.append(loss.item())

            accuracy = correct / total
            accuracy_list.append(accuracy)
            #print(loss_list)
            #print(accuracy_list)
        # Validation Loop
        """
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
        """

    #print(f"Average Accuracy: {np.mean(results):.4f}")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    with open(log_file_path, 'a') as f:
      f.write("\n\n\n")
      f.write(f"Training completed at {current_datetime}\n")
      f.write("=================================\n")
      #f.write(f"{description}\n")
      #f.write(f"Accuracy obtained = {np.mean(results):.4f}\n")
      f.write(f"{epoch_list}\n")
      f.write(f"{loss_list}\n")
      f.write(f"{accuracy_list}\n")
    save_checkpoint(checkpoint)
    return results


results = kfold_cross_validation(num_rounds,  batch_size,model)
