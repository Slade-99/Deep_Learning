####### Evalutions #######
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.EdgeViT.model import model
#from Implementations.Proper_Practice.Final_Testing.Model.Custom_Architecture.sparse_att import model
#from Implementations.Proper_Practice.Final_Testing.Model.MobileNet_V2.model import model
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.MobileViT_S.model_gradcam import model
from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.MobileNet_V2.model_new import model
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.Swin.model_new import model

#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.CVT.model_gradcam import model
from During_Thesis.Implementations.Preprocessings.Private_Dataset_Preprocessings.Testing_prepare_private_dataset import test_dataloader ,eval_transforms
#from Implementations.Proper_Practice.Final_Testing.Utils.utils import save_checkpoint,load_checkpoint
import torch
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file_path = "/home/azwad/Works/Deep_Learning/During_Thesis/Implementations/Proper_Practice/Final_Testing/Results_Data/LeViT/testing_metric.txt"
description = "LeViT"
model_path = '/home/azwad/Works/Model_Weights/MobileNet_V2.pth.tar'
class_names = ['normal', 'abnormal' , 'pneumonia' ]




learning_rate = 0.0001

global_model = model
checkpoint = torch.load(model_path)
global_model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
global_model.load_state_dict(checkpoint['state_dict'])
#print(global_model.levit_model.classifier.linear.bias)
global_model = model.to(device=device)

global_model.eval()
all_labels = []
all_preds = []
all_probs = []  # To store probabilities for ROC and AUROC


with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = global_model(images)
        probs = torch.softmax(outputs, dim=1)  # Get probabilities
        #print(probs)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')



# Compute AUROC (Area Under the ROC Curve) for multi-class
# Binarize the labels for multi-class
all_labels_binarized = label_binarize(all_labels, classes=np.unique(all_labels))
auroc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')



sample_image, _ = test_dataloader.dataset[0]
sample_image_pil = to_pil_image(sample_image)
start_time = time.time()
sample_image_transformed = eval_transforms(sample_image_pil)  # Apply transforms
output = global_model(sample_image_transformed.unsqueeze(0).to(device))
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
    #f.write(f"  - FLOPs: {flops.total() / 1e9:.2f} GFLOPs\n")
    f.write(f"  - Latency: {latency:.4f} s\n\n\n\n")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names, 
    annot_kws={"size": 14, "fontweight": "bold"}  # Increase annotation font size and make it bold
)
plt.xlabel('Predicted', fontsize=14, fontweight="bold")
plt.ylabel('True', fontsize=14, fontweight="bold")
#plt.title('Confusion Matrix')
plt.xticks(fontsize=14, fontweight="bold", rotation=0)
plt.yticks(fontsize=14, fontweight="bold", rotation=90)
plt.show()
#plt.savefig('confusion_matrix_LeViT.png')




# Ensure that all_labels_binarized and all_probs are numpy arrays
all_labels_binarized = np.array(all_labels_binarized)  # Convert to numpy array
all_probs = np.array(all_probs)  # Convert to numpy array

# Plot ROC curve for each class
fpr, tpr, _ = roc_curve(all_labels_binarized.ravel(), all_probs.ravel())
# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(len(np.unique(all_labels))):
    fpr_i, tpr_i, _ = roc_curve(all_labels_binarized[:, i], all_probs[:, i])
    auc_i = auc(fpr_i, tpr_i)
    plt.plot(fpr_i, tpr_i, label=f'Class {i} (AUC = {auc_i:.2f})')

# Plot random chance line
#plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

# Add labels and title
plt.xlabel('False Positive Rate', fontsize=14, fontweight="bold")
plt.ylabel('True Positive Rate', fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
#plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'})

# Show the plot
plt.show()