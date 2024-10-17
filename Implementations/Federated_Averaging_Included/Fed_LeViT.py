"""
Image_Size -> 224
Params_Count -> 8,443,267


"""


import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import torchvision
from torch.utils.data import Subset
import random
from sklearn.metrics import accuracy_score
import numpy as np
#from transformers import CvTModel, CvTConfig
from torch import nn
from torchvision import transforms
from torchinfo import summary
import os
import random
from transformers import LevitConfig, LevitModel
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis


model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomLeViT(nn.Module):
    def __init__(self , num_labels):
        super(CustomLeViT, self).__init__()

        # Load the CvT model configuration
        config = LevitConfig(num_channels=1 , num_labels = 3)
        self.levit = LevitModel(config  )
        self.classifier = nn.Linear(384, num_labels)


    def forward(self, x):
        output = self.levit(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits
        
    

class_names = ['normal', 'covid', 'pneumonia']
# Initialize the CvT model
pretrained_levit = CustomLeViT(3).to(device)
#summary(pretrained_levit, input_size=(1, 1,224, 224)) 
# Print a summary using torchinfo

""""
summary(model=pretrained_levit, 
        input_size=(16, 1, 224, 224),  # 1 channel for grayscale images
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
"""


pretrained_levit_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])



train_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/train'
test_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/test'
NUM_WORKERS = os.cpu_count()
def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int=NUM_WORKERS):
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=pretrained_levit_transforms,
    batch_size=16
)

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_levit.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()


num_clients = 2
num_rounds = 2

clients_data = [[] for _ in range(num_clients)]

clients_data = [[] for _ in range(num_clients)]

# Step 1: Distribute the dataset among clients
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

distribute_data(train_dataloader_pretrained)


num_epochs_per_client = 1
for round_idx in range(num_rounds):
    print(f"Round {round_idx + 1}/{num_rounds}")

    # Step 2: Randomly select 50% of clients
    selected_clients = random.sample(range(num_clients), num_clients // 2)
    client_models = []

    # Step 3: Each selected client trains on its local dataset
    for client_idx in selected_clients:
        print(f"Client {client_idx + 1}/{num_clients} training...")

        local_model = pretrained_levit  # Copy the global model
        local_model.train()  # Set model to training mode
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)  # Define optimizer for the client
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Create DataLoader for the client's local dataset
        client_dataloader = torch.utils.data.DataLoader(clients_data[client_idx], batch_size=32, shuffle=True)

        # Train for a specified number of epochs
        for epoch in range(num_epochs_per_client):
            print(f"Client {client_idx + 1} - Epoch {epoch + 1}/{num_epochs_per_client}")
            for images, labels in tqdm(client_dataloader, desc=f"Training Client {client_idx + 1}", unit="batch"):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()  # Clear gradients
                outputs = local_model(images)  # Get model predictions
                loss = loss_fn(outputs, labels)  # Calculate loss

                # Backward pass and optimization
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update parameters

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)  # Get predicted classes
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average loss and accuracy for the client
        avg_loss = running_loss / len(client_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Client [{client_idx + 1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        client_models.append(local_model.state_dict())  # Store the client's model parameters

    # Step 4: Federated Averaging
    global_model_state_dict = pretrained_levit.state_dict()
    for key in global_model_state_dict.keys():
        # Ensure tensors are in float format before averaging
        global_model_state_dict[key] = torch.mean(
            torch.stack([client_model[key].float() for client_model in client_models]), dim=0
        )

    # Update the global model with the averaged parameters
    pretrained_levit.load_state_dict(global_model_state_dict)

    # Save the updated global model
    model_path = os.path.join(model_dir, f"levit_round_{round_idx + 1}.pth")
    torch.save(pretrained_levit.state_dict(), model_path)
    print(f"Global model saved to {model_path}")

print("Federated Learning Training Complete!")







pretrained_levit.eval()  # Set the model to evaluation mode
all_labels = []
all_preds = []

with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_dataloader_pretrained:
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_levit(images)
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Classification Report
print(classification_report(all_labels, all_preds, target_names=class_names))
flops = FlopCountAnalysis(pretrained_levit, torch.randn(1, 1, 224, 224).to(device))
print(f"FLOPs: {flops.total()}")
# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
