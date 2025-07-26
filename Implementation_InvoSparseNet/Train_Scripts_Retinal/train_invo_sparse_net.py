# Imports
import torch
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim, nn  
from torchvision.models import mobilenet_v3_small,convnext_tiny,efficientnet_v2_s,shufflenet_v2_x0_5,squeezenet1_0,densenet121
from Implementation_Phase.Models.InvoSparseNet.model_v2 import invo_sparse_net
from Implementation_Phase.Models.MobileVitV2.model import mobilevitv2
from Implementation_Phase.Models.CVT.model import cvt
from Implementation_Phase.Models.EdgeVitXXS.model import edgevit
from Implementation_Phase.Models.SwinV2.model import swinv2
from torch.utils.data import DataLoader
#from Implementation_Phase.Testing.test import eval_data
from tqdm import tqdm
from torchsummary import summary
import time
import os
import numpy as np
from PIL import Image
import cv2




sleeping = False
load = False
current_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_list = ["Private_CXR","Retinal_OCT","Skin_Cancer_ISIC","Br34H","PC"]
models_list = {"invo_sparse_net":invo_sparse_net , "mobilenet_v3_small":mobilenet_v3_small, "convnext_tiny":convnext_tiny,"efficientnet_v2_s":efficientnet_v2_s,"shufflenet_v2_x0_5":shufflenet_v2_x0_5,"squeezeNet1_0":squeezenet1_0}
dataset = "Retinal_OCT"
selected_model = "invo_sparse_net"
#model = models_list[selected_model]
params_count = 0
log_dir = "/home/azwad/Works/Deep_Learning/Implementation_Phase/Evaluation_Data/"+dataset+"/"+selected_model+".txt"
eval_file_path = log_dir+"_evaluation.txt"
os.makedirs(os.path.dirname(log_dir), exist_ok=True)
log_file = open(log_dir,"a")
log_file.write(f"Training {selected_model} on {dataset} at {current_time}\n")
log_file.write(f"-----------------------------------------------\n\n\n")



#### Hyperparameters ####
in_channels = 1
num_classes = 8
learning_rate = 0.00001
batch_size = 16
num_epochs = 50
#########################




#### Dataset Preparation ######################################################################################

data_dir = "/mnt/hdd/Datasets/Medical/" + dataset + "/"
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
model = mobilenet_v3_small(num_classes=3)
model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
model = model.to(device)

model = convnext_tiny(num_classes=3)
model.features[0][0] = nn.Conv2d(1, 96, kernel_size=3, stride=2, padding=1, bias=False)
model = model.to(device)

model = densenet121(num_classes=3,)
model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(device)

model = efficientnet_v2_s(num_classes=3,).to(device)
model.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)

model = shufflenet_v2_x0_5(num_classes=3).to(device)
model.conv1[0] = nn.Conv2d(1,24,3,2,1,bias=False)


model = squeezenet1_0(num_classes=3).to(device)
model.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))


model = mobilevitv2.to(device)

model = edgevit.to(device)
"""


model = invo_sparse_net.to(device)

#summary(model, input_size =(1,224,224))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
################################


def save_checkpoint(state):
    save_dir = "/mnt/hdd/Trained_Weights/"+dataset+"/"+selected_model+"/"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    filename=save_dir+selected_model+"_"+str(current_time)
    filename = filename+".pth.tar"
    print("=>Saving checkpoint")
    torch.save(state,filename)
    

def load_checkpoint(checkpoint,optimizer):
  print("=>Loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
    
if load:
    model_path = '/mnt/hdd/Trained_Weights/NIH/invo_sparse_net/tbc2.pth.tar'
    checkpoint = torch.load(model_path)
    load_checkpoint(checkpoint,optimizer)


    
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
    model = model.to(device)
    for epoch in range(1,num_epochs+1):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            
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

                
        
        
        
        if(epoch%5==0):
            train_accuracy = check_accuracy(train_loader, model)
            val_accuracy = check_accuracy(val_loader, model)
            log_file.write(f"Results on epoch {epoch}\n")
            log_file.write("------------------------------\n")
            log_file.write(f"Accuracy on training set: {train_accuracy*100:.2f}\n")
            log_file.write(f"Accuracy on validation set: { val_accuracy* 100:.2f}%\n")
            print(f"Accuracy on training set: {train_accuracy*100:.2f}\n")
            print(f"Accuracy on validation set: {check_accuracy(val_loader, model) * 100:.2f}%\n")
            log_file.write("\n\n")
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)
            
            




    log_file.write(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
    log_file.write("\n\n\n\n")
    log_file.close()
    
train(model)


#all_labels,all_preds,all_probs = eval_data(eval_file_path ,model,test_loader,data_transforms)