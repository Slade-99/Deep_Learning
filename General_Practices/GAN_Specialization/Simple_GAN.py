import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter




class Discriminator(nn.Module):
    
    def __init__(self,img_dim):
        
        super().__init__()
        
        
        self.disc = nn.Sequential(
            
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid(),
            
        )
    
    def forward(self,x):
        return self.disc(x)
    



class Generator(nn.Module):
    
    def __init__(self,z_dim,img_dim):
        
        super().__inti__()
        
        self.gen = nn.Sequential(
            
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh(),
            
        )
        
        
    def forward(self,x):
        
        return self.gen(x)
    


## Hyperparameter

device = "cuda" if torch.cuda.is_available() else 'cpu'
lr = 3e-4
z_dim = 64
img_dim = 28*28*1

batch_size = 32
num_epochs = 50




disc = Discriminator(img_dim).to(device)
gen =  Generator(z_dim,img_dim).to(device)
fixed_noise = torch.randn((batch_size,z_dim)).to(device)




transforms = transforms.Compose(
    
    [transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))]
    
)

dataset = datasets.MNIST(root="", download=True)
loader = DataLoader(dataset , batch_size=batch_size, shuffule = True)
opt_disc = optim.Adam(disc.parameters(), lr =lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)


writer_fake = SummaryWriter(f"")
writer_real = SummaryWriter(f"")
step = 0




for epoch in range(num_epochs):
    for batch_idx , ( real , _) in enumerate(loader):
        real = real.view(-1,784).to(device)
        batch_size = real.shape[0]
        
        
    noise = torch.randn(batch_size,z_dim).to(device)
    fake = gen(noise)
    disc_real = disc(real).view(-1)
    