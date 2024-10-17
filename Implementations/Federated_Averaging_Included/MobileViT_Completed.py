### Imports ###
from skimage import exposure, filters
from skimage.filters import median
from skimage.morphology import disk
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import time
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import torchvision
from torch.utils.data import Subset
import random
from sklearn.metrics import accuracy_score
import numpy as np
from torch import nn
from torchvision import transforms
from torchinfo import summary
import os
import random
from transformers import MobileViTV2Config, MobileViTV2Model
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix , f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
import cv2


#### Necessary Variables ####
model_dir = "saved_models"
log_file_path = "training_logs.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(model_dir, exist_ok=True)
class_names = ['normal', 'abnormal', 'pneumonia']
train_dir = '/home/azwad/Datasets/Shortened/train'
test_dir = '/home/azwad/Datasets/Shortened/test'
NUM_WORKERS = os.cpu_count()
num_clients = 2
num_rounds = 2
num_epochs_per_client = 1
log_file_name = "Federated Learning Training Log for MobileViT\n"






# File to save statistics
with open(log_file_path, 'w') as f:
    f.write(log_file_name)
    f.write("=================================\n")










class MobileViT(nn.Module):
    def __init__(self , num_labels):
        super(MobileViT, self).__init__()

        # Load the CvT model configuration
        config = MobileViTV2Config(num_channels=1 , num_labels = 3)
        self.MobileViT = MobileViTV2Model(config)
        self.classifier = nn.Linear(512, num_labels)


    def forward(self, x):
        output = self.MobileViT(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits

pretrained_mobileViT = MobileViT(3).to(device)
    


####### Dataset Preparation ########
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)
    
class MedianBlurTransform:
    def __call__(self, img):
        img = np.array(img)
        img = cv2.medianBlur(img, 5) 
        return Image.fromarray(img) 

pretrained_mobileViT_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    #CLAHETransform(),
    #MedianBlurTransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int=NUM_WORKERS):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes
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

train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=pretrained_mobileViT_transforms,
    batch_size=16
)








# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_mobileViT.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()



















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

distribute_data(train_dataloader_pretrained)













####  Federated Learning Training Loop ####
for round_idx in range(num_rounds):
    print(f"Round {round_idx + 1}/{num_rounds}")
    selected_clients = random.sample(range(num_clients), num_clients // 2)
    client_models = []
    avg_accuracy = 0
    avg_loss = 0
    for client_idx in selected_clients:
        print(f"Client {client_idx + 1}/{num_clients} training...")

        local_model = pretrained_mobileViT
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
                
                loss = loss_fn(outputs, labels)
                
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

    global_model_state_dict = pretrained_mobileViT.state_dict()
    for key in global_model_state_dict.keys():
        global_model_state_dict[key] = torch.mean(torch.stack([client_model[key].float() for client_model in client_models]), dim=0)

    pretrained_mobileViT.load_state_dict(global_model_state_dict)

    # Save global model
    model_path = os.path.join(model_dir, f"mobileViT_round_{round_idx + 1}.pth")
    torch.save(pretrained_mobileViT.state_dict(), model_path)
    print(f"Global model saved to {model_path}")

    # Log round details
    with open(log_file_path, 'a') as f:
        f.write(f"Round {round_idx + 1}:\n")
        f.write(f"  - Loss: {avg_loss:.4f}\n")
        f.write(f"  - Accuracy: {avg_accuracy:.4f}\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
####### Evalutions #######
pretrained_mobileViT.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_dataloader_pretrained:
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_mobileViT(images)
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
sample_image, _ = test_dataloader_pretrained.dataset[0]
sample_image_pil = to_pil_image(sample_image)
start_time = time.time()
sample_image_transformed = pretrained_mobileViT_transforms(sample_image_pil)  # Apply transforms
output = pretrained_mobileViT(sample_image_transformed.unsqueeze(0).to(device))
end_time = time.time()
latency = end_time - start_time
flops = FlopCountAnalysis(pretrained_mobileViT, torch.randn(1, 1, 224, 224).to(device))
summary(pretrained_mobileViT, input_size=(1, 1, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], depth=3)
with open(log_file_path, 'a') as f:
    f.write("\nFinal Evaluation:\n")
    f.write(f"  - Accuracy: {accuracy:.4f}\n")
    f.write(f"  - F1 Score: {f1:.4f}\n")
    f.write(f"  - Precision: {precision:.4f}\n")
    f.write(f"  - Recall: {recall:.4f}\n")
    f.write(f"  - FLOPs: {flops.total() / 1e9:.2f} GFLOPs\n")
    f.write(f"  - Latency: {latency:.4f} s\n")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_MobileViT.png')
