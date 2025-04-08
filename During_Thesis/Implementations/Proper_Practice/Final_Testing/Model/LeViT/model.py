import torch
import torch.nn as nn
from transformers import LevitConfig, LevitModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MYMODEL(nn.Module):
    def __init__(self , num_labels):
        super(MYMODEL, self).__init__()

        config = LevitConfig(num_channels=1 , num_labels = 3)
        self.archi = LevitModel(config)
        self.classifier = nn.Linear(384, num_labels)


    def forward(self, x):
        outputs = self.archi(x)

        pooler_output = outputs.pooler_output  

        logits = self.classifier(pooler_output)  
        return logits




model = MYMODEL(3).to(device)






#total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(trainable_params)