from model import device,optimizer,model
import torch


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


def save_checkpoint(state, filename="/content/drive/MyDrive/Architecture Weights/RedNet/RedNet.pth.tar"):
  print("=>Saving checkpoint")
  torch.save(state,filename)


def load_checkpoint(checkpoint):
  print("=>Loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])