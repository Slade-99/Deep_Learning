# Imports
import torch
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim, nn  
from Implementation_Phase.Ablation_Study.Model_Variants.var8 import invo_sparse_net
from torch.utils.data import DataLoader
from tqdm import tqdm  
from torchsummary import summary
import time
import os
import numpy as np
from PIL import Image
import cv2

sleeping = True
current_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_list = ["Private_CXR","Retinal_OCT","Skin_Cancer_ISIC","Br35H","PC"]
dataset ="Private_CXR"
selected_model = "invo_sparse_net"
model = invo_sparse_net
params_count = 0
log_dir = "/home/azwad/Works/Deep_Learning/Implementation_Phase/Ablation_Study/Results/logs.txt"
os.makedirs(os.path.dirname(log_dir), exist_ok=True)
log_file = open(log_dir,"a")
log_file.write(f"Training on variant 8 with 2Inv 2Conv 2Inv with Inv kernel of size 9  layer arrangement 2.282M parameters\n\n")
log_file.write(f"-----------------------------------------------\n\n\n")



#### Hyperparameters ####
in_channels = 1  
num_classes = 2
learning_rate = 0.0001
batch_size = 16
num_epochs = 25
#########################




#### Dataset Preparation ######################################################################################

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

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#####################################################################################################################






###### Model Preparation #######
"""
model = mobilenet_v2(num_classes=3)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model = model.to(device)
"""
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
################################


def save_checkpoint(state):
    save_dir = "/mnt/hdd/Trained_Weights/"+dataset+"/"+selected_model+"/"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    filename=save_dir+selected_model+"_"+str(current_time)
    filename = filename+".pth.tar"
    print("=>Saving checkpoint")
    torch.save(state,filename)
    
    

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# Train Network
def train(model):
    for epoch in range(1,num_epochs+1):
        for batch_idx, (data, targets) in enumerate(train_loader):
            
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            if(batch_idx%100 == 0 and sleeping):
                time.sleep(3)
        
        if(epoch%5==0):
            log_file.write(f"Results on epoch {epoch}\n")
            log_file.write("------------------------------\n")
            log_file.write(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}\n")
            log_file.write(f"Accuracy on validation set: {check_accuracy(val_loader, model) * 100:.2f}%\n")
            log_file.write("\n\n")
            #checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            #save_checkpoint(checkpoint)
            
            




    log_file.write(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
    log_file.write("\n\n\n\n")
    log_file.close()
    
train(model)