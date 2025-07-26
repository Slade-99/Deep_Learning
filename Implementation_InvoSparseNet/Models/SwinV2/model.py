import torch
import torch.nn as nn
from transformers import Swinv2Config, Swinv2Model
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SWINV2(nn.Module):
    def __init__(self):
        super(SWINV2, self).__init__()

        # Load the CvT model configuration
        configuration = Swinv2Config(num_channels=1 , num_labels = 3)
        self.Swin = Swinv2Model(configuration)
        self.classifier = nn.Linear(768, 3)


    def forward(self, x):
        output = self.Swin(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape

        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits

swinv2 = SWINV2().to(device)
# Total parameters (including trainable and non-trainable)
total_params = sum(p.numel() for p in swinv2.parameters())

# Only trainable parameters
trainable_params = sum(p.numel() for p in swinv2.parameters() if p.requires_grad)

#print(f"Total Parameters: {total_params}")
#print(f"Trainable Parameters: {trainable_params}")