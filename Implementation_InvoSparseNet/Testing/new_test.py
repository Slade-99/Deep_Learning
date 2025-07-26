####### Evalutions #######
import torch 
import torch.nn as nn
from Implementation_Phase.Models.InvoSparseNet.model_v2 import invo_sparse_net
from Implementation_Phase.Models.SwinV2.model import swinv2
from Implementation_Phase.Models.MobileVitV2.model import mobilevitv2
from torchvision.models import mobilenet_v3_small,convnext_tiny,efficientnet_v2_s,shufflenet_v2_x0_5,swin_v2_t,squeezenet1_0
import torch
from Implementation_Phase.Models.CVT.model import cvt
from Implementation_Phase.Models.EdgeVitXXS.model import edgevit
from torchsummary import summary
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix , f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import time
import seaborn as sns
from torch import optim
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import torchvision.datasets as datasets 
from PIL import Image
import os
import torch.nn.functional as F
from thop import profile



dataset = "Retinal_OCT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
learning_rate = learning_rate = 0.00005

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
class_names = train_dataset.classes
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = invo_sparse_net.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
def load_checkpoint(checkpoint,optimizer):
  print("=>Loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  #optimizer.load_state_dict(checkpoint['optimizer'])



all_labels = []
all_preds = []
all_probs = []
model.eval()

def eval_data(log_file_path,model,test_loader,data_transforms):


    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())



    class_names = sorted(np.unique(all_labels))  # Ensures consistency



    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    sample_image, _ = test_loader.dataset[0]
    sample_image_pil = to_pil_image(sample_image)
    start_time = time.time()
    sample_image_transformed = data_transforms["test"](sample_image_pil)  # Apply transforms
    output = model(sample_image_transformed.unsqueeze(0).to(device))
    end_time = time.time()
    latency = end_time - start_time










    #flops = FlopCountAnalysis(global_model, torch.randn(1, 1, 224, 224).to(device))
    #summary(global_model, input_size=(1, 1, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], depth=3)

    with open(log_file_path, 'a') as f:
        f.write("\nFinal Evaluation :\n")
        f.write("\n{description} :\n")
        f.write(f"  - Accuracy: {accuracy:.4f}\n")
        f.write(f"  - F1 Score: {f1:.4f}\n")
        f.write(f"  - Precision: {precision:.4f}\n")
        f.write(f"  - Recall: {recall:.4f}\n")
        
        f.write(f"  - Latency: {latency:.4f} s\n\n\n\n")
    
    
    print(all_preds)
    
    




    


num_classes = 8
mobilevit_v2 = mobilevitv2(num_classes)
edgevit = edgevit.to(device)
mobilenet_v3 = mobilenet_v3_small(num_classes)
mobilenet_v3.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
mobilenet_v3 = mobilenet_v3.to(device)
print(mobilenet_v3)
shufflenet = shufflenet_v2_x0_5(num_classes).to(device)
shufflenet.conv1[0] = nn.Conv2d(1,24,3,2,1,bias=False)
shufflenet = shufflenet.to(device)
squeezenet = squeezenet1_0(num_classes).to(device)
squeezenet.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
squeezenet = squeezenet.to(device)



model_list = [mobilenet_v3,invo_sparse_net,mobilevit_v2,edgevit,shufflenet,squeezenet]
name_dict = {
        invo_sparse_net:"invo_sparse_net",
    mobilevit_v2:"mobilevit_v2",
    edgevit:"edgevit",
    mobilenet_v3:"mobilenet_v3",
    shufflenet:"shufflenet",
    squeezenet:"squeezenet"
}
weight_dict = {
    invo_sparse_net:"/mnt/hdd/Trained_Weights/Retinal_OCT/invo_sparse_net/invo_sparse_net_1751140969.8933876.pth.tar",
    mobilevit_v2:"/mnt/hdd/Trained_Weights/Retinal_OCT/mobilevitv2/mobilevitv2_1751152796.4679778.pth.tar",
    edgevit:"/mnt/hdd/Trained_Weights/Retinal_OCT/edgevitv2/edgevitv2_1751135788.8646576.pth.tar",
    mobilenet_v3:"/mnt/hdd/Trained_Weights/Retinal_OCT/mobilenet_v3_small/mobilenet_v3_small_1751148302.2583132.pth.tar",
    shufflenet:"/mnt/hdd/Trained_Weights/Retinal_OCT/shufflenet_v2_x0_5/shufflenet_v2_x0_5_1751157607.9839706.pth.tar",
    squeezenet:"/mnt/hdd/Trained_Weights/Retinal_OCT/squeezenet1_0/squeezenet1_0_1751162173.1505914.pth.tar"
}

for model in model_list:
    load = True
    model_name = name_dict[model]
    model = model.to(device)
    log_file = '/home/azwad/Works/Deep_Learning/Implementation_Phase/Evaluation_Data/'
    log_file_path = log_file + dataset+"/"+model_name+"_evaluation.txt"

    if load:
        
        model_path = weight_dict[model]
        checkpoint = torch.load(model_path)
        load_checkpoint(checkpoint,optimizer)


    eval_data(log_file_path,model,test_loader,data_transforms)