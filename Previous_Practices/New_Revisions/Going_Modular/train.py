import os 
import torch 
import torch.nn as nn
from torch import optim
from torchvision import transforms
import data_setup,model_builder,engine,utils
from timeit import default_timer as timer 
from torchinfo import summary



### Hyperparameters 
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
CORES = os.cpu_count()



### Setup directories
train_dir = "/mnt/hdd/Datasets/Torchvision_Datasets/pizza_steak_sushi/train"
test_dir  =  "/mnt/hdd/Datasets/Torchvision_Datasets/pizza_steak_sushi/test"


## Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"



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



## Create a model
model = model_builder.TinyVGG(input_shape=3 , hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)




start_time = timer()
engine.train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,optimizer=optimizer,criterion=criterion,epochs=10,device=device)
end_time = timer()

print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

utils.save_model("TinyVGG.pth",model,"/mnt/hdd/Research_works/Trial1_weights")