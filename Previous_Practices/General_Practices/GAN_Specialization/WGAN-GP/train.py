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
from utils import gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-5
batch_size = 64
img_size = 64
img_channels =3
noise_dim = 100
num_classes = None
gen_embedding = 100
filters_critic = 64
filters_gen = 64
num_epochs = 5
critic_iterations = 5
lambda_gp = 10



transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize( [0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)]),

    ]
)




dataset = datasets.ImageFolder(root="/home/azwad/Works/Deep_Learning/dataset/CelebA/img_align_celeba/" , transform = transforms)
loader = DataLoader(dataset , batch_size=batch_size , shuffle=True)
gen = Generator(noise_dim, img_channels , filters_gen,num_classes,img_size,gen_embedding).to(device)
critic = Discriminator(img_channels,filters_critic,num_classes,img_size).to(device)
initialize_weights(gen)
initialize_weights(critic)


opt_gen = optim.Adam(gen.parameters(), lr = lr , betas = (0.0 ,0.9))
opt_critic = optim.Adam(critic.parameters(), lr = lr , betas = (0.0 ,0.9))


fixed_noise = torch.randn(32,noise_dim,1,1).to(device)

writer_real = SummaryWriter(f"/home/azwad/Works/Deep_Learning/dataset/GAN_Generated/WGAN_GP_CelebA/real/")
writer_fake = SummaryWriter(f"/home/azwad/Works/Deep_Learning/dataset/GAN_Generated/WGAN_GP_CelebA/fake/")
step = 0


gen.train()
critic.train()


for epoch in range(num_epochs):
    for batch_idx, (real, labels) in enumerate(loader):
        
        
        
        for _ in range(critic_iterations):
            real = real.to(device)
            labels = labels.to(device)
            noise = torch.randn((batch_size,noise_dim,1,1)).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real , labels).reshape(-1)
            critic_fake = critic(fake , labels).reshape(-1)
            gp = gradient_penalty(critic,labels,real,fake,device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp*gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            

        output = critic(fake,labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        ### Print Losses and Print to tensorborad ###
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch/num_epochs}] Batch [{batch_idx}/{len(loader)}] \
                  Loss Critic: {loss_critic:.4f} , Loss Generator: {loss_gen:.4f}"
                )
            
            with torch.no_grad():
                
                fake = gen(noise, labels)
                
                img_grid_real = torchvision.utils.make_grid(
                    real[:32] , normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32],normalize=True
                )

                writer_real.add_image("Real" , img_grid_real , global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
            step+=1