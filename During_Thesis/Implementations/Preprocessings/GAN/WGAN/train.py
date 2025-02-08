
import os
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty
from data_loader import train_dataloader

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4    # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_CLASSES = 3
GEN_EMBEDDING = 100
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 224
FEATURES_GEN = 224
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10




SAVE_DIR = "./generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

"""
transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)
"""

# If you train on MNIST, remember to set channels_img to 1
#dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

# comment mnist above and uncomment below if train on Celeb
#dataset = datasets.ImageFolder(root="/home/azwad/Datasets/Benchmark_Dataset/Filtered", transform=transforms)
dataloader = train_dataloader
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES,IMAGE_SIZE,GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC,NUM_CLASSES,IMAGE_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE , betas=(0.0 , 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE  , betas=(0.0 , 0.9) )


fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.to(device)
        labels = labels.to(device)

        
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise,labels)
            critic_real = critic(real,labels).reshape(-1)
            critic_fake = critic(fake,labels).reshape(-1)
            gp = gradient_penalty(critic,labels,real,fake,device=device)
            loss_critic =    (  -(torch.mean(critic_real) - torch.mean(critic_fake))  + LAMBDA_GP*gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph = True)
            opt_critic.step()
            
            

                
                
        output = critic(fake,labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        


        # Save images every 20 batches
        if batch_idx % 20 == 0:
            noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
            with torch.no_grad():
                fake = gen(noise,labels)
                
                # Save real images
                save_image(real[:32], os.path.join(SAVE_DIR, f"real_epoch{epoch}_batch{batch_idx}.png"), normalize=True)
                # Save fake images
                save_image(fake[:32], os.path.join(SAVE_DIR, f"fake_epoch{epoch}_batch{batch_idx}.png"), normalize=True)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
            Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )
