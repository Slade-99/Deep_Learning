import torch
import torch.nn as nn
from transformers import CvtConfig, CvtModel
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CvT(nn.Module):
    def __init__(self):
        super(CvT, self).__init__()

        # Load the CvT model configuration
        configuration = CvtConfig(num_channels=1 , num_labels = 3)
        self.CvT = CvtModel(configuration)
        self.classifier = nn.Linear(384, 3)


    def forward(self, x):
        output = self.CvT(x)
        pooler_output = output.cls_token_value  
        pooler_output = pooler_output.squeeze(1)
        logits = self.classifier(pooler_output)  
        return logits

model = CvT().to(device)
print(model)
#summary(model, input_size =(1,224,224))
#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)