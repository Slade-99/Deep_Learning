import torch
import torch.nn as nn
from transformers import MobileViTV2Config, MobileViTV2Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### 4.389M ###

class MobileViTV2(nn.Module):
    def __init__(self , num_labels):
        super(MobileViTV2, self).__init__()

        # Load the CvT model configuration
        config = MobileViTV2Config(num_channels=1 , num_labels = 3 , hidden_act='gelu')
        self.MobileViT = MobileViTV2Model(config)
        self.classifier = nn.Linear(512, num_labels)


    def forward(self, x):
        output = self.MobileViT(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits



def mobilevitv2(classes):
    return MobileViTV2(classes).to(device)


# Total parameters (including trainable and non-trainable)
#total_params = sum(p.numel() for p in mobilevitv2.parameters())

# Only trainable parameters
#trainable_params = sum(p.numel() for p in mobilevitv2.parameters() if p.requires_grad)

#print(f"Total Parameters: {total_params}")
#print(f"Trainable Parameters: {trainable_params}")