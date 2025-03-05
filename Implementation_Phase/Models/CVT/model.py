import torch
import torch.nn as nn
from transformers import CvtConfig, CvtModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#### 19.6M ####

class CvT(nn.Module):
    def __init__(self):
        super(CvT, self).__init__()

        # Load the CvT model configuration
        configuration = CvtConfig(num_channels=1 , num_labels = 3)
        self.CvT = CvtModel(configuration)
        self.classifier = nn.Linear(384, 11)


    def forward(self, x):
        output = self.CvT(x)
        pooler_output = output.cls_token_value  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        pooler_output = pooler_output.squeeze(1)
        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits

cvt = CvT().to(device)

# Total parameters (including trainable and non-trainable)
total_params = sum(p.numel() for p in cvt.parameters())

# Only trainable parameters
trainable_params = sum(p.numel() for p in cvt.parameters() if p.requires_grad)

#print(f"Total Parameters: {total_params}")
#print(f"Trainable Parameters: {trainable_params}")