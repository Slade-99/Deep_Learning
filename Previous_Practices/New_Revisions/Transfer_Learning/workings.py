from Going_Modular import engine,data_setup,model_builder,utils,plotting,predict_and_plot
from timeit import default_timer as timer 
import os
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torchvision
import random
from pathlib import Path


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
#weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=None)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(in_features=1280,out_features=3,bias=True)
model.classifier
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)




start_time = timer()
results = engine.train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,optimizer=optimizer,criterion=criterion,epochs=NUM_EPOCHS,device=device)
end_time = timer()

print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

utils.save_model("EfficientNetB0.pth",model,"/mnt/hdd/Research_works/Trial1_weights")


plotting.plot_loss_curves(results)



num_images_to_plot  = 10
test_image_path_list = list(Path(test_dir).resolve().glob("*/*.jpg"))
test_images_samples = random.sample(population=test_image_path_list, k = num_images_to_plot)



for image_path in test_images_samples:
    
    predict_and_plot.pred_and_plot_image(model=model,image_path=image_path,class_names=class_names,image_size=(224,224))