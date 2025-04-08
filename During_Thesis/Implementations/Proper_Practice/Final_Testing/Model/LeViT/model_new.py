import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor

# Define the number of classes in your dataset
NUM_CLASSES = 3
# Choose a specific LeViT model configuration name from Hugging Face Hub.
# This defines the ARCHITECTURE SIZE (e.g., embedding dim, layers),
# but we won't load its pre-trained weights.
# Example: 'facebook/levit-128s', 'facebook/levit-256', etc.
MODEL_CONFIG_NAME = "facebook/levit-128s" # You can change this configuration name

class LeViTForImageClassificationCustomScratch(nn.Module):
    """
    A custom module wrapping a LeViT model architecture from Hugging Face,
    initialized with RANDOM WEIGHTS (trained from scratch), and configured
    for a specified number of classes.
    """
    def __init__(self, model_config_name: str,num_channels:int , num_labels: int):
        """
        Args:
            model_config_name (str): The name of the LeViT model configuration
                                     on Hugging Face Hub (e.g., 'facebook/levit-128s').
                                     This defines the architecture.
            num_labels (int): The number of output classes for the classification task.
        """
        super().__init__()
        self.num_labels = num_labels
        self.model_config_name = model_config_name
        self.gradients = None
        self.gradCAM = False

        # 1. Load the configuration associated with the chosen model name.
        #    This contains the architectural parameters (layers, dimensions, etc.)
        print(f"Loading configuration for '{self.model_config_name}'...")
        config = AutoConfig.from_pretrained(self.model_config_name)

        # 2. Update the configuration to match your specific task.
        #    Most importantly, set the number of labels for the classifier head.
        config.num_labels = self.num_labels
        config.num_channels = num_channels
        # You could potentially modify other config settings here if needed,
        # though it's often best to start with the defaults for the chosen size.
        # config.dropout_rate = 0.1 # Example modification (if applicable)

        print(f"Initializing LeViT model architecture from configuration "
              f"with {self.num_labels} output classes (random weights)...")
        # 3. Initialize the model architecture using the modified configuration.
        #    `from_config` ensures the model structure is built according to
        #    the `config` object, but all weights are initialized randomly.
        #    No pre-trained weights are downloaded or used.
        self.levit_model = AutoModelForImageClassification.from_config(config)
        self.target_layer = self.levit_model.levit.patch_embeddings.embedding_layer_4.convolution
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)
        #print(self.target_layer)
        # Optional: Print the model structure to verify
        # print("Model Structure (Randomly Initialized):")
        # print(self.levit_model)
        # print("\nClassifier Head:")
        # print(self.levit_model.classifier)

    def forward_hook(self, module, input, output):

        self.feature_maps = output

    def backward_hook(self, module, grad_in, grad_out):

        self.gradients = grad_out[0]
    
    
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the randomly initialized model.

        Args:
            pixel_values (torch.Tensor): A batch of preprocessed image tensors.
                                         Shape typically (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: The output logits from the classification head.
                          Shape (batch_size, num_labels).
        """
        # Pass the preprocessed pixel values through the underlying LeViT model
        outputs = self.levit_model(pixel_values=pixel_values)
        # Extract the logits
        logits = outputs.logits
        return logits

    def get_activations_gradient(self):

        return self.gradients

    def get_activations(self, x):

        _ = self.forward(x)  
        return self.feature_maps



# --- Example Usage ---






# 1. Instantiate the custom model class (will have random weights)
print(f"Creating model based on '{MODEL_CONFIG_NAME}' config for {NUM_CLASSES} classes...")
custom_scratch_levit_model = LeViTForImageClassificationCustomScratch(
    model_config_name=MODEL_CONFIG_NAME,
    num_channels=1,
    num_labels=NUM_CLASSES
)
print("Model created with random weights.")

# 2. Load the corresponding image processor
#    Even when training from scratch, you need the processor to format
#    the input images correctly (e.g., size, normalization) for the
#    specific LeViT architecture defined by the configuration.
processor = AutoImageProcessor.from_pretrained(MODEL_CONFIG_NAME)
print(f"Image processor for '{MODEL_CONFIG_NAME}' loaded (for input formatting).")




# 3. Example: Create dummy input data (replace with your actual preprocessed data)
try:
    if hasattr(processor, 'size') and isinstance(processor.size, dict) and 'height' in processor.size and 'width' in processor.size:
         height = processor.size["height"]
         width = processor.size["width"]
    elif hasattr(processor, 'size') and isinstance(processor.size, int):
         height = width = processor.size
    else:
         print("Could not automatically determine input size from processor, assuming 224x224.")
         height, width = 224, 224
except AttributeError:
    print("Could not automatically determine input size from processor, assuming 224x224.")
    height, width = 224, 224

batch_size = 4
num_channels = 1
dummy_pixel_values = torch.randn(batch_size, num_channels, height, width)
print(f"\nCreated dummy input tensor with shape: {dummy_pixel_values.shape}")

# 4. Perform a forward pass (usually within a training loop)
#    Set model to train mode if you were calculating gradients
custom_scratch_levit_model.eval() # Set to evaluation mode for this inference example
with torch.no_grad(): # Disable gradient calculations for inference
    logits = custom_scratch_levit_model(pixel_values=dummy_pixel_values)

print(f"Output logits shape: {logits.shape}") # Should be (batch_size, NUM_CLASSES)
print("Example output logits (first item in batch - from random weights):")
print(logits[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_scratch_levit_model.to(device)
