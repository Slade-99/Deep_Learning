import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm 

### Device ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#### Hyper-parameters ####
IMG_SIZE = 784  # (28x28)
NUM_CLASSES  = 10 
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1
NUM_LAYERS = 4



### Load Data ###


train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)







### Architecture ###
class NN(nn.Module):
    
    def __init__(self,input_size , num_classes, n ):
        super(NN,self).__init__()
        self.n = n
        self.fc = []
        self.fc.append(nn.Linear(input_size,2**(n+2)))
        self.fc.append(nn.ReLU())
        
        
        for i in range(n,2,-1):
            self.fc.append(nn.Linear(2**(i+2),2**(i+1)))
            self.fc.append(nn.ReLU())
 
        self.fc.append(nn.Linear(16,num_classes))
        self.fc = nn.Sequential(*self.fc)
        
        
        
    def forward(self,x):
        return self.fc(x)




### Initialize the Model ###
model = NN(IMG_SIZE,NUM_CLASSES,NUM_LAYERS)


"""
model = NN(IMG_SIZE,NUM_CLASSES)
x = torch.randn(64,784)
print(model(x).shape)
"""




### Loss Function and Optimizer ###
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters() , LEARNING_RATE)


        
        
        
        

####  Training Loop ####

for epoch in range(EPOCHS):
    for batch_idx , (data,targets) in enumerate(train_loader):
        
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        if epoch==0 and batch_idx ==0:
            print(f"Data : {data.shape}")
            print(f"Targets : {targets.shape}")
        
        data = data.reshape(data.shape[0],-1)

        if epoch==0 and batch_idx ==0:
            print(data.shape)
            print(targets.shape)
        
        scores = model(data)
        loss = criterion(scores,targets)


        if epoch==0 and batch_idx ==0:
            print(f"Scores : {scores.shape}")
            print(f"Loss: {loss}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Done with epoch {epoch+1} ")
        
        
        
        









### Evalutaions #####

def check_accuracy(loader,model):
    
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
        
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)
            
            
            scores = model(x)
            _,predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
            
        print(f"Got {num_correct} / {num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f} ")
        

    model.train()
    

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)