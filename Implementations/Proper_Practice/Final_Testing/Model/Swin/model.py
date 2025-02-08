import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SWIN(nn.Module):
    def __init__(self):
        super(SWIN, self).__init__()

        # Load the CvT model configuration
        configuration = SwinConfig(num_channels=1 , num_labels = 3)
        self.Swin = SwinModel(configuration)
        self.classifier = nn.Linear(768, 3)


    def forward(self, x):
        output = self.Swin(x)
        pooler_output = output.pooler_output  # Get the CLS token output
         # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape

        logits = self.classifier(pooler_output)  # Pass through the classification head
        return logits

model = SWIN().to(device)
#summary(model, input_size =(1,224,224))