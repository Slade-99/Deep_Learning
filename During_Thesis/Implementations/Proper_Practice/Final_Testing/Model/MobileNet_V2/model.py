import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchsummary import summary


model = models.mobilenet_v2(pretrained=True).to(device)

# Modify the first convolutional layer to accept 1-channel input (grayscale image)
# The first layer is a Conv2d layer, which takes 3 input channels by default
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

# Modify the last fully connected layer to output 3 classes (instead of 1000 for ImageNet)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

# Print the model architecture to verify changes
#print(model)
#summary(model, input_size =(1,224,224))
#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)