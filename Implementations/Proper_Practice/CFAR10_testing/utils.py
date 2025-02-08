from Model.model import model
import torch





def save_checkpoint(state, name):
    filename="/home/azwad/Works/Deep_Learning/Implementations/architecture_weights/"
    filename = filename+name+".pth.tar"
    print("=>Saving checkpoint")
    torch.save(state,filename)


def load_checkpoint(checkpoint,optimizer):
  print("=>Loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])