import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
weight = 0.7
bias = 0.3


start = 0
end = 15
step = 0.2

X = torch.arange(start,end,step)
y = weight*X + bias


train_split = int(0.8*len(X))
val_split = int(0.9*len(X))
X_train , y_train = X[:train_split] , y[:train_split]
X_val , y_val = X[train_split:val_split] , y[train_split:val_split]
X_test , y_test = X[val_split:], y[val_split:]



def plot_predictions(train_data=X_train,train_label=y_train,test_data=X_test,test_label=y_test,prediction=None):
    
    plt.figure(figsize=(10,7))
    
    plt.scatter(train_data,train_label,c='b',s=4,label='Training Data')
    plt.scatter(test_data,test_label,c='g',s=4,label='Test Data')
    
    if (prediction!=None):
        plt.scatter(test_data,prediction,c='r',s=4,label="Predictions")
        
        
    plt.legend(prop={"size":14})
    plt.show()
    



class Linear_Regression(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,requires_grad=True,dtype=torch.float))
        
        
    def forward(self,x):
        
        return self.weights*x + self.bias
        


torch.manual_seed(41)


model = Linear_Regression()




criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)


epochs = 200
epoch_count = []
train_loss = []
test_loss = []
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    if epoch%10 ==0:
        model.eval()
        with torch.inference_mode():
            preds = model(X_test)
        

        
        loss_test = criterion(preds,y_test)
        test_loss.append(loss_test)
        train_loss.append(loss)
        epoch_count.append(epoch)
        model.train()
    



print(model.state_dict())



plt.plot(epoch_count,np.array(torch.tensor(train_loss).numpy()),label="Train_Loss")
plt.plot(epoch_count,np.array(torch.tensor(test_loss).numpy()),label="Test_Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()



checkpoint = { 'epoch': 10,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}
torch.save(checkpoint,"Linear_Regression_2.pth.tar")



received = torch.load("Linear_Regression_2.pth.tar")
model.load_state_dict(received['model_state_dict'])
optimizer.load_state_dict(received['optimizer_state_dict'])
epoch = received['epoch']
loss = received['loss']


