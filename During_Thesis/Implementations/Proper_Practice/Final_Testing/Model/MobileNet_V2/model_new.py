import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchsummary import summary

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.gradients = None  
        self.feature_maps = None  


        self.mobilenetv2 = models.mobilenet_v2(pretrained=True).to(device)


        self.mobilenetv2.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


        self.mobilenetv2.classifier[1] = nn.Linear(self.mobilenetv2.classifier[1].in_features, 3)


        self.target_layer = self.mobilenetv2.features[-1]


        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):

        self.feature_maps = output

    def backward_hook(self, module, grad_in, grad_out):

        self.gradients = grad_out[0]

    def forward(self, x):

        return self.mobilenetv2(x)

    def get_activations_gradient(self):

        return self.gradients

    def get_activations(self, x):

        _ = self.forward(x)  
        return self.feature_maps  


model = MobileNetV2().to(device)