
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

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DISC = 1e-4   # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 16
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 64
FEATURES_GEN = 64



SAVE_DIR = "./generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)


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


# If you train on MNIST, remember to set channels_img to 1
#dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

# comment mnist above and uncomment below if train on Celeb
dataset = datasets.ImageFolder(root="/home/azwad/Datasets/Benchmark_Dataset/Filtered", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"Implementations/Preprocessings/GAN/Deep Convolutional GAN/logs/real")
writer_fake = SummaryWriter(f"Implementations/Preprocessings/GAN/Deep Convolutional GAN/logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Save images every 20 batches
        if batch_idx % 20 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                
                # Save real images
                save_image(real[:32], os.path.join(SAVE_DIR, f"real_epoch{epoch}_batch{batch_idx}.png"), normalize=True)
                # Save fake images
                save_image(fake[:32], os.path.join(SAVE_DIR, f"fake_epoch{epoch}_batch{batch_idx}.png"), normalize=True)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
            Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
        )
