import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128 * 16 * 16),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Unflatten(1, (128, 16, 16)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.model(z)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)




class CGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def forward(self, z):
        return self.generator(z)




def train_cgan(generator, discriminator, cgan, dataloader, num_epochs=50, batch_size=64, lr=0.0002, beta1=0.5):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            outputs = discriminator(imgs.to(device))
            d_loss_real = criterion(outputs, real_labels)
            
            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = generator(z)
            outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
        
        if (epoch+1) % 10 == 0:
            save_generated_images(generator, epoch+1)





def save_generated_images(generator, epoch, num_samples=16):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 100).to(device)
        gen_imgs = generator(z).cpu().numpy()
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]
        
        fig, axs = plt.subplots(4, 4, figsize=(4, 4))
        cnt = 0
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(np.transpose(gen_imgs[cnt], (1, 2, 0)), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.savefig(f'cgan_images_epoch_{epoch}.png')
        plt.close()
    generator.train()




class CustomDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = plt.imread(self.img_paths[idx])
        img = np.expand_dims(img, axis=0)  # Assuming grayscale images
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



dataset = CustomDataset(img_paths=['path_to_image1', 'path_to_image2'], transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)
cgan = CGAN(generator, discriminator).to(device)

train_cgan(generator, discriminator, cgan, dataloader, num_epochs=50)
