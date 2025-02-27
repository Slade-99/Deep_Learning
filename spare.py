import torch

torch.cuda.empty_cache()  # Clears unused memory
torch.cuda.ipc_collect()  # Helps with inter-process memory management
