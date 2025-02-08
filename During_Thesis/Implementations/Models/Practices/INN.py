import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

class Involution(nn.Module):
    def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio):
        super(Involution, self).__init__()
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        # Kernel generation layers
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction_ratio, kernel_size * kernel_size * group_number, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Scale the height and width with respect to the strides.
        new_height = H // self.stride
        new_width = W // self.stride
        
        if self.stride > 1:
            kernel_input = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=(self.stride - 1) // 2)
        else:
            kernel_input = x

        kernel = self.kernel_gen(kernel_input)

        # Debugging: Print kernel shape
        print(f'Kernel shape: {kernel.shape}')

        kernel = kernel.view(B, self.kernel_size * self.kernel_size, new_height, new_width, self.group_number)

        input_patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size // 2)
        input_patches = input_patches.view(B, C // self.group_number, self.kernel_size * self.kernel_size, new_height, new_width, self.group_number)

        # Debugging: Print input_patches shape
        print(f'Input patches shape: {input_patches.shape}')

        output = kernel * input_patches

        output = output.sum(dim=2)  # (B, H', W', C // G, G)

        output = output.view(B, -1, new_height, new_width)

        # Debugging: Print output shape
        print(f'Output shape: {output.shape}')

        return output, kernel

class InvolutionModel(nn.Module):
    def __init__(self):
        super(InvolutionModel, self).__init__()
        self.inv1 = Involution(channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1)  # 1 channel
        self.inv2 = Involution(channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1)  # 1 channel
        self.inv3 = Involution(channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1)  # 1 channel
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 64, 64)  # Adjust based on final output size
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x, _ = self.inv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x, _ = self.inv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x, _ = self.inv3(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


# Example dataset preparation
# Assuming train_data and test_data are numpy arrays of shape (num_samples, 32, 32)
train_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/train'
test_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/test'

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int):
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

INN_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])
# Setup dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=INN_transforms,
    batch_size=16
)

# Initialize the model, loss function, and optimizer
model = InvolutionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Optionally, add validation here