import torch
import torch.nn as nn
from transformers import MobileViTConfig, MobileViTModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MobileViT(nn.Module):
    def __init__(self, num_labels):
        super(MobileViT, self).__init__()
        self.use_gradCAM = False
        self.gradients = None  # To store gradients

        # Load the MobileViT model
        config = MobileViTConfig(num_channels=1, num_labels=num_labels)
        self.MobileViT = MobileViTModel(config)
        
        # Choose a feature map layer for Grad-CAM (e.g., last conv block)
        self.target_layer = self.MobileViT.encoder.layer[-1]  # Last transformer block

        # Fully connected classifier
        self.classifier = nn.Linear(640, num_labels)

    def activations_hook(self, grad):
        """ Store gradients during backpropagation """
        self.gradients = grad

    def forward(self, x):
        """ Forward pass with Grad-CAM support """
        outputs = self.MobileViT(x)

        # Get the feature maps from the target layer
        feature_maps = outputs.last_hidden_state  # Extract activations

        if self.use_gradCAM:
            feature_maps.register_hook(self.activations_hook)  # Hook for gradients

        pooler_output = outputs.pooler_output  # CLS token output
        logits = self.classifier(pooler_output)

        if self.use_gradCAM:
            return logits  # Return feature maps for Grad-CAM
        else:
            return logits

    def get_activations_gradient(self):
        """ Retrieve stored gradients """
        return self.gradients

    def get_activations(self, x):
        """ Forward pass to get activations from target layer """
        with torch.no_grad():
            outputs = self.MobileViT(x)
        return outputs.last_hidden_state  # Feature maps


model = MobileViT(3).to(device)
