import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from Implementation_Phase.Ablation_Study.Model_Variants.var17 import prepare_architecture
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time


sleeping = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset ="Private_CXR"
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, dropout_rate, num_layers):
        super(CustomModel, self).__init__()
        layers = []
        in_features = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_sizes[i]))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_sizes[i]

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
    
    
data_dir = "/mnt/hdd/Datasets/" + dataset + "/"
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)


# Data transformations with augmentation for training
data_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "test": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}


# Load datasets
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=data_transforms["val"])
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=data_transforms["test"])


    
    








def objective(trial):
    # Define hyperparameters to tune
    
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8,16])
    
    activation = trial.suggest_categorical("activation", ["relu", "tanh","gelu","silu"])
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD","RMSProp"])

    # Load dataset (modify this based on your data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = prepare_architecture(activation=activation).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Choose optimizer
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1)
    elif optimizer_type == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=1)

    # Train model for a few epochs
    for epoch in range(5):  # Keep it small for tuning
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        time.sleep(7)
                

    # Evaluate on validation set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)  # Return validation loss

# Run hyperparameter optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)
