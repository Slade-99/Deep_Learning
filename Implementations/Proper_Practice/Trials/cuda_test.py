import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    # Get GPU details
    gpu_name = torch.cuda.get_device_name(0)
    num_gpus = torch.cuda.device_count()
    allocated_memory = torch.cuda.memory_allocated(0)  # Memory allocated on GPU 0
    reserved_memory = torch.cuda.memory_reserved(0)    # Reserved memory on GPU 0

    print(f"CUDA Available: {cuda_available}")
    print(f"Number of GPUs available: {num_gpus}")
    print(f"Using GPU: {gpu_name}")
    print(f"Memory allocated on GPU 0: {allocated_memory / 1024**2:.2f} MB")
    print(f"Memory reserved on GPU 0: {reserved_memory / 1024**2:.2f} MB")
else:
    print("CUDA is not available, using CPU.")
