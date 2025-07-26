import torch 
import torch.utils.tensorboard
import torchvision
import matplotlib.pyplot as plt 
from Going_Modular import engine,data_setup,model_builder,utils,plotting,predict_and_plot
from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict,List,Tuple
import torch
import os
from tqdm.auto import tqdm
from Going_Modular.engine import train_step,test_step
from torch import optim
CORES = os.cpu_count()
BATCH_SIZE = 32
LEARNING_RATE = 0.001
### Setup directories
train_dir = "/mnt/hdd/Datasets/Torchvision_Datasets/pizza_steak_sushi/train"
test_dir  =  "/mnt/hdd/Datasets/Torchvision_Datasets/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter(log_dir="Experiments")
model = torchvision.models.efficientnet_b0(weights=None)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(in_features=1280,out_features=3,bias=True)
model.classifier
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)



### Data Transforms

train_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
test_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor(), 
])


train_dataloader , test_dataloader , class_names = data_setup.create_dataloaders(train_dir=train_dir,test_dir=test_dir,train_transforms=train_transforms,test_transforms=test_transforms,num_worker=CORES,batch_size=BATCH_SIZE)





def train(model:torch.nn.Module , train_dataloader:torch.utils.data.DataLoader , test_dataloader: torch.utils.data.DataLoader , optimizer:torch.optim.Optimizer, criterion:torch.nn.Module, epochs:int = 5, device=device , writer: torch.utils.tensorboard.writer.SummaryWriter = None):
    
    
    results = {"train_loss":[] ,  "train_accuracy":[] , "test_loss":[], "test_accuracy":[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss , train_acc = train_step(model=model , dataloader=train_dataloader , criterion=criterion , optimizer=optimizer,device=device)
        test_loss , test_acc = test_step(model=model , dataloader=test_dataloader , criterion=criterion , optimizer=optimizer,device=device)
        
        
        print(f"Epoch: {epoch} | Training Loss : {train_loss:.4f}  | Test Loss : {test_loss:.4f}  | Train Accuracy: {train_acc:.4f}  |  Test Accuracy {test_acc:.4f}" )
        
        results["train_accuracy"].append(train_acc)
        results["test_accuracy"].append(test_acc)
        results["test_loss"].append(test_loss)
        results["train_loss"].append(train_loss)
        

        if writer:
            writer.add_scalars(main_tag="Loss" , tag_scalar_dict={"train_loss":train_loss , "Test_loss":test_loss}, global_step=epoch)
            
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"training_acc":train_acc , "test_acc":test_acc} , global_step=epoch)
            
            
            writer.add_graph(model=model,input_to_model=torch.rand(32,3,224,224).to(device))
        
            writer.close()
        
        else:
            pass
    
    return results




results = train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,optimizer=optimizer,criterion=criterion,epochs=5,device=device)
