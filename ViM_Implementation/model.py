from vision_mamba import vim_base,vim_small,vim_tiny
import torch 
import torch.nn as nn
from torchinfo import summary 
device = "cuda" if torch.cuda.is_available() else "cpu"

model_vim_tiny = vim_tiny(pretrained=True)

model_vim_tiny = model_vim_tiny.to(device)



model_vim_tiny.head = nn.Linear(192, 10).to(device)
#summary(model,input_size=(1,3,224,224))
#x = torch.rand(16,3,224,224).to(device='cuda')
#print(model(x).shape)