"""
Contains the functions for training and testing a PyTorch model.
"""


from typing import Dict,List,Tuple
import torch
from tqdm.auto import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model: torch.nn.Module ,  dataloader: torch.utils.data.DataLoader , criterion:torch.nn.Module , optimizer : torch.optim.Optimizer , device=device):
    
    model.train()
    model = model.to(device)
    train_loss , train_accuracy = 0 , 0
    
    for batch , ( images , labels) in tqdm(enumerate(dataloader)):
        
        images = images.to(device)
        labels = labels.to(device)
         
        outputs = model(images)
         
        loss = criterion(outputs,labels)
        optimizer.zero_grad()

         
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_class = torch.argmax(torch.softmax(outputs,dim=1) , dim=1)
        train_accuracy += (pred_class==labels).sum().item()/len(outputs)
    
    
    train_loss = train_loss/len(dataloader)
    train_accuracy = train_accuracy/len(dataloader)
    return train_loss, train_accuracy
    




def test_step(model: torch.nn.Module ,  dataloader: torch.utils.data.DataLoader , criterion:torch.nn.Module , optimizer : torch.optim.Optimizer, device=device):
    
    model.eval()
    model = model.to(device)
    test_loss , test_accuracy = 0 , 0
    
    with torch.inference_mode():
    
        for batch , ( images , labels) in tqdm(enumerate(dataloader)):
            
            images = images.to(device)
            
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs,labels)
            test_loss += loss.item()
            pred_class = torch.argmax(torch.softmax(outputs,dim=1) , dim=1)
            test_accuracy += (pred_class==labels).sum().item()/len(outputs)
    
    
    test_loss = test_loss/len(dataloader)
    test_accuracy = test_accuracy/len(dataloader)
    return test_loss , test_accuracy





def train(model:torch.nn.Module , train_dataloader:torch.utils.data.DataLoader , test_dataloader: torch.utils.data.DataLoader , optimizer:torch.optim.Optimizer, criterion:torch.nn.Module, epochs:int = 5, device=device ):
    
    
    results = {"train_loss":[] ,  "train_accuracy":[] , "test_loss":[], "test_accuracy":[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss , train_acc = train_step(model=model , dataloader=train_dataloader , criterion=criterion , optimizer=optimizer,device=device)
        test_loss , test_acc = test_step(model=model , dataloader=test_dataloader , criterion=criterion , optimizer=optimizer,device=device)
        
        
        print(f"Epoch: {epoch} | Training Loss : {train_loss:.4f}  | Test Loss : {test_loss:.4f}  | Train Accuracy: {train_acc:.4f}  |  Test Accuracy {test_acc:.4f}" )
        
        results["train_accuracy"].append(train_acc)
        results["test_accuracy"].append(test_acc)
        results["test_loss"].append(test_loss)
        results["train_loss"].append(train_loss)
        
        
    
    return results