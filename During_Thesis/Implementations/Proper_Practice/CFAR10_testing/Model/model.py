import torch
import torch.nn as nn
from Implementations.Proper_Practice.CFAR10_testing.Model.inn_dwconv import custom_resnet
from Implementations.Proper_Practice.CFAR10_testing.Model.sparse_att import sparse_attn_model
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






backbone = custom_resnet

model = nn.Sequential(
    backbone,
    sparse_attn_model,

)

model = model.to(device=device)


x = torch.rand(1,3,32,32).to(device)
print(model(x).shape)
summary(model, input_size =(3,32,32))




