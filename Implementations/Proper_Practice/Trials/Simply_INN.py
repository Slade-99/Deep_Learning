# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules  # Gives easier dataset managment by creating mini batches etc.
from data_loader import train_dataloader, test_dataloader
from tqdm import tqdm  # For nice progress bar!



# Hyperparameters
in_channels = 1
num_classes = 3
learning_rate = 3e-4 # karpathy's constant
batch_size = 2
num_epochs = 1
# Load Data

def save_checkpoint(state, filename="/home/azwad/Works/DL_Models_Checkpoint/Simple_INN.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])





# Simple CNN
class INN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(INN, self).__init__()
        
        
        
        
        self.inv1 = Involution(in_channels, 3, kernel_size=(3, 3), stride=(1, 1), groups=3)
        
        
        
        self.inv2 = Involution(3, 3, kernel_size=(3, 3), stride=(1, 1), groups=3)
        
        
        
        self.inv3 = Involution(3, 3, kernel_size=(3, 3), stride=(1, 1), groups=3)
        
        
        
        self.fc1 = nn.Linear(158700, num_classes)

    def forward(self, x):
        x = F.relu(self.inv1(x))
        
        x = F.relu(self.inv2(x))
        x = F.relu(self.inv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x





class Involution(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, groups, reduce_ratio=3, dilation=(1, 1), padding=(3, 3), bias=False):
        super().__init__()
        self.bias = bias
        self.padding = padding
        self.dilation = dilation
        self.reduce_ratio = reduce_ratio
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.output_ch = output_ch
        self.input_ch = input_ch
        self.init_mapping = nn.Conv2d(in_channels=self.input_ch, out_channels=self.output_ch, kernel_size=(1, 1), stride=(1, 1), bias=self.bias) if self.input_ch != self.output_ch else nn.Identity()
        self.reduce_mapping = nn.Conv2d(in_channels=self.input_ch, out_channels=self.output_ch // self.reduce_ratio, kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.span_mapping = nn.Conv2d(in_channels=self.output_ch // self.reduce_ratio, out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups, kernel_size=(1, 1), stride=(1, 1),
                                      bias=self.bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        self.pooling = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.sigma = nn.Sequential(
            nn.BatchNorm2d(num_features=self.output_ch // self.reduce_ratio, momentum=0.3), nn.ReLU())

    def forward(self, inputs):
        batch_size, _, in_height, in_width = inputs.shape
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                     // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                    // self.stride[1] + 1

        unfolded_inputs = self.unfold(self.init_mapping(inputs))
        inputs = F.adaptive_avg_pool2d(inputs,(out_height,out_width))
        unfolded_inputs = unfolded_inputs.view(batch_size, self.groups, self.output_ch // self.groups, self.kernel_size[0] * self.kernel_size[1], out_height, out_width)

        kernel = self.pooling(self.span_mapping(self.sigma(self.reduce_mapping((inputs)))))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        output = (kernel * unfolded_inputs).sum(dim=3)

        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
        return output

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






train_loader = train_dataloader
test_loader = test_dataloader








# Initialize network
model = INN(in_channels=in_channels, num_classes=num_classes).to(device)





criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)








### Load checkpoint ####
#load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)





# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

