### Training DCGAN


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator , Generator , initialize_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 2e-4
batch_size = 128
img_size = 64
img_channels =1
noise_dim = 100
filters_disc = 64
filters_gen = 64
num_epochs = 50


transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.normalize( [0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)]),

    ]
)


dataset = datasets.MNIST(root="" , train=True , transform = transforms,download = True)
loader = DataLoader(dataset , batch_size=batch_size , shuffle=True)
gen = Generator(noise_dim, img_channels , filters_gen).to(device)
disc = Discriminator(img_channels,filters_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)


opt_gen = optim.Adam(gen.parameters(), lr = lr , betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(), lr = lr , betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,noise_dim,1,1).to(device)

writer_real = SummaryWriter(f"")
writer_fake = SummaryWriter(f"")
step = 0


gen.train()
disc.train()


for epoch in range(num_epochs):
    for batch_idx, (real,_) in enumerate(loader):

        real = real.to(device)
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        noise = torch.randn((batch_size,noise_dim,1,1)).to(device)
        disc_fake = disc(noise).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real+loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()


        output = disc(noise).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        ### Print Losses and Print to tensorborad ###


