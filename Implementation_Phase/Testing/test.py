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


load = True
model_name = "invosparseneth"
dataset = "NIH"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file = '/home/azwad/Works/Deep_Learning/Implementation_Phase/Evaluation_Data/'
log_file_path = log_file + dataset+"/"+model_name+"_evaluation.txt"
batch_size = 16
learning_rate = learning_rate = 0.00005

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

if load:
    model_path = '/mnt/hdd/Trained_Weights/NIH/invo_sparse_net/invo_sparse_net_1741048105.95246.pth.tar'
    checkpoint = torch.load(model_path)
    load_checkpoint(checkpoint,optimizer)

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



    input_tensor = torch.randn(1, 1, 224, 224).to(device)

    # Calculate FLOPs and Parameters
    flops, params = profile(model, inputs=(input_tensor,))

    # Convert FLOPs to GFLOPs
    gflops = flops / 1e9
    params_million = params / 1e6  # Convert to million

    # Print results
    print(f"GFLOPs: {gflops:.2f} GFLOPs")
    print(f"Parameters: {params_million:.2f} Million")






    #flops = FlopCountAnalysis(global_model, torch.randn(1, 1, 224, 224).to(device))
    #summary(global_model, input_size=(1, 1, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], depth=3)

    with open(log_file_path, 'a') as f:
        f.write("\nFinal Evaluation :\n")
        f.write("\n{description} :\n")
        f.write(f"  - Accuracy: {accuracy:.4f}\n")
        f.write(f"  - F1 Score: {f1:.4f}\n")
        f.write(f"  - Precision: {precision:.4f}\n")
        f.write(f"  - Recall: {recall:.4f}\n")
        f.write(f"GFLOPs: {gflops:.2f} GFLOPs\n")
        f.write(f"  - Latency: {latency:.4f} s\n\n\n\n")
    
    
    print(all_preds)
eval_data(log_file_path,model,test_loader,data_transforms)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
cm_path = log_file+dataset+"/"+model_name+"_confusion_matrix.png"
#plt.savefig(cm_path)




import numpy as np


# Convert to numpy arrays for further processing
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Binarize the labels for multi-class classification
class_names = sorted(np.unique(all_labels))  # Ensure sorted class order
all_labels_binarized = label_binarize(all_labels, classes=class_names)

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
class_names = train_dataset.classes
for i in range(len(class_names)):
    # Compute ROC curve for the i-th class
    fpr_i, tpr_i, _ = roc_curve(all_labels_binarized[:, i], all_probs[:, i])
    auc_i = auc(fpr_i, tpr_i)
    
    # Plot the ROC curve for the i-th class
    plt.plot(fpr_i, tpr_i, label=f'Class {class_names[i]} (AUC = {auc_i:.2f})')

# Add the random chance line (diagonal)
#plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

# Add labels and title to the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')

# Save the plot to a file
roc_path = "roc_curve.png"
#plt.savefig(roc_path)

# Show the plot
plt.show()
