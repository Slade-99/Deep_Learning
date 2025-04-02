import torch
import torch.nn as nn
from transformers import MobileViTConfig, MobileViTModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MobileViT(nn.Module):
    def __init__(self , num_labels):
        super(MobileViT, self).__init__()

        # Load the CvT model configuration
        config = MobileViTConfig(num_channels=1 , num_labels = 3)
        self.MobileViT = MobileViTModel(config)
        self.classifier = nn.Linear(640, num_labels)


    def forward(self, x):
        output = self.MobileViT(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits

model = MobileViT(3).to(device)
#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)