try:
    from torch.utils.tensorboard import SummaryWriter
    print("SummaryWriter is installed!")
except ImportError:
    print("SummaryWriter is not installed.")