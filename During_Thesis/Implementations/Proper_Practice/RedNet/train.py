import time
import torch
from preprocessing import train_dataloader, test_dataloader
from model import model,device,criterion,optimizer,num_epochs
from tqdm import tqdm
from utils import check_accuracy,save_checkpoint,load_checkpoint






LOAD = False


if LOAD:
    
    model = load_checkpoint(torch.load("/content/drive/MyDrive/Architecture Weights/ResNet/ResNet.pth.tar"))
    





# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        time.sleep(5)

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    #save_checkpoint(checkpoint)

    print(f"Accuracy on training set: {check_accuracy(test_dataloader, model)*100:.2f}")