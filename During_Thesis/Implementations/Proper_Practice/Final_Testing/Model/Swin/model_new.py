import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SWIN(nn.Module):
    def __init__(self):
        super(SWIN, self).__init__()

        # Load the Swin Transformer model configuration
        configuration = SwinConfig(num_channels=1, num_labels=3, output_hidden_states=True)
        
        self.Swin = SwinModel(configuration)  # Load the Swin Transformer model
        self.classifier = nn.Linear(768, 3)  # Adjust hidden size based on Swin model output

        # Hooks for Grad-CAM
        self.activations = None
        self.gradients = None
        self.target_layer = None
        # Hooking into the last layer of the last Swin block
        self.target_layer = self.Swin.encoder.layers[2].blocks[3]
        print(self.target_layer)
        self.hook_layers()

    def hook_layers(self):
        """Register forward and backward hooks to extract feature maps and gradients."""
        def forward_hook(module, input, output):
            self.activations = output  # Save feature map activations

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]  # Save gradients

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def get_activations_gradient(self):

        return self.gradients

    def get_activations(self, x):

        _ = self.forward(x)  
        return self.target_layer 
    
    
    def forward(self, x):
        output = self.Swin(x)  # Forward pass through Swin Transformer
        pooler_output = output.pooler_output  # CLS token output for classification
        logits = self.classifier(pooler_output)  # Pass through classification head
        return logits


model = SWIN().to(device)
#print(model)
#print(model)
#summary(model, input_size =(1,224,224))
#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)