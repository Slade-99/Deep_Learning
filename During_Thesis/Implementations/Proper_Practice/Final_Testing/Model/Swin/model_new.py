import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SWIN(nn.Module):
    def __init__(self):
        super(SWIN, self).__init__()
        self.gradients = None
        self.use_gradCAM = False
        # Load the Swin Transformer model configuration
        configuration = SwinConfig(num_channels=1, num_labels=3, output_hidden_states=True)
        
        self.Swin = SwinModel(configuration)  # Load the Swin Transformer model
        self.classifier = nn.Linear(768, 3)  # Adjust hidden size based on Swin model output

        # Hooks for Grad-CAM

        # Hooking into the last layer of the last Swin block

    def activations_hook(self, grad):
        """ Store gradients during backpropagation """
        self.gradients = grad


    
    def forward(self, x):
        output = self.Swin(x)  # Forward pass through Swin Transformer
        feature_maps = output.last_hidden_state
        
        if self.use_gradCAM:
            feature_maps.register_hook(self.activations_hook)  # Hook for gradients
        
        pooler_output = output.pooler_output  # CLS token output for classification
        logits = self.classifier(pooler_output)  # Pass through classification head
        return logits

    def get_activations_gradient(self):
        """ Retrieve stored gradients """
        return self.gradients

    def get_activations(self, x):
        """ Forward pass to get activations from target layer """
        with torch.no_grad():
            outputs = self.Swin(x)
        return outputs.last_hidden_state  # Feature maps
    
model = SWIN().to(device)

#print(model)
#summary(model, input_size =(1,224,224))
#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)