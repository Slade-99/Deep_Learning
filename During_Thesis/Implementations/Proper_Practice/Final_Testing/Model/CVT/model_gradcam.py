import torch
import torch.nn as nn
from transformers import CvtConfig, CvtModel
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CvT(nn.Module):
    def __init__(self):
        super(CvT, self).__init__()
        self.gradients = None  # To store gradients

        # Load the CvT model
        configuration = CvtConfig(num_channels=1, num_labels=3)
        self.CvT = CvtModel(configuration)

        # Fully connected classifier
        self.classifier = nn.Linear(384, 3)

    def activations_hook(self, grad):
        """ Store gradients during backpropagation """
        self.gradients = grad

    def forward(self, x):
        """ Forward pass with Grad-CAM support """
        outputs = self.CvT(x)


        feature_maps = outputs.last_hidden_state  
        #print(feature_maps)

        feature_maps.register_hook(self.activations_hook)  # Hook for gradients

        pooler_output = outputs.cls_token_value  
        pooler_output = pooler_output.squeeze(1)
        logits = self.classifier(pooler_output)  
        return logits

    def get_activations_gradient(self):
        """ Retrieve stored gradients """
        return self.gradients

    def get_activations(self, x):
        """ Forward pass to get activations from target layer """
        with torch.no_grad():
            outputs = self.CvT(x)
        return outputs.last_hidden_state  # Feature maps



model = CvT().to(device)
