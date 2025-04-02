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

        # Get the feature maps from the target layer (last convolutional layer)
        feature_maps = outputs.last_hidden_state  # Extract activations
        print(feature_maps)
        feature_maps.requires_grad_()
        feature_maps.register_hook(self.activations_hook)  # Hook for gradients
        print(self.gradients)
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

x = torch.rand(1, 1, 224, 224).to(device)  # Input tensor
logits = model(x)  # Forward pass

# 🔥 **Trigger Gradients Computation**
target_class = logits.argmax(-1).item()  # Predicted class
print(target_class)
logits[:, target_class].backward(retain_graph=True)  # Compute gradients

# 🔍 **Check if gradients are stored**
print("Gradients: ", model.get_activations_gradient())
#summary(model, input_size =(1,224,224))
#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)