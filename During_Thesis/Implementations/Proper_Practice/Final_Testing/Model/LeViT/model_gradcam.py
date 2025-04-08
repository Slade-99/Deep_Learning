import torch
import torch.nn as nn
from transformers import LevitConfig, LevitModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LeViT(nn.Module):
    def __init__(self , num_labels):
        super(LeViT, self).__init__()
        self.gradCAM = False
        self.gradients = None  # To store gradients
        # Load the CvT model configuration
        config = LevitConfig(num_channels=1 , num_labels = 3)
        self.LeViT = LevitModel(config)
        self.classifier = nn.Linear(384, num_labels)





    def activations_hook(self, grad):
        """ Store gradients during backpropagation """
        self.gradients = grad

    def forward(self, x):
        outputs = self.LeViT(x)
          

        if self.gradCAM:
            feature_maps = outputs.last_hidden_state
            feature_maps.requires_grad_()
            feature_maps.register_hook(self.activations_hook)  

        pooler_output = outputs.pooler_output  

        logits = self.classifier(pooler_output)  
        return logits






    def get_activations_gradient(self):
        """ Retrieve stored gradients """
        return self.gradients




    def get_activations(self, x):
        """ Forward pass to get activations from target layer """
        with torch.no_grad():
            outputs = self.LeViT(x)
        return outputs.last_hidden_state  # Feature maps






model = LeViT(3).to(device)






#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)