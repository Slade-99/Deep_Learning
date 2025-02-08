import torch
import torch.nn as nn
from transformers import LevitConfig, LevitModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LeViT(nn.Module):
    def __init__(self , num_labels):
        super(LeViT, self).__init__()

        # Load the CvT model configuration
        config = LevitConfig(num_channels=1 , num_labels = 3)
        self.MobileViT = LevitModel(config)
        self.classifier = nn.Linear(384, num_labels)


    def forward(self, x):
        output = self.MobileViT(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits

model = LeViT(3).to(device)