import torch
import torch.nn as nn
from transformers import CvtConfig, CvtModel
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CvT(nn.Module):
    def __init__(self):
        super(CvT, self).__init__()
        self.gradients = None  # To store gradients

        # Load the CvT model
        configuration = CvtConfig(num_channels=1, num_labels=3)
        self.CvT = CvtModel(configuration)

        # Fully connected classifier
        self.classifier = nn.Linear(384, 3)
        self.target_layer = self.CvT.encoder.stages[-1].layers[-1].attention.attention.convolution_projection_value.convolution_projection.convolution


        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)


    def forward_hook(self, module, input, output):

        self.feature_maps = output

    def backward_hook(self, module, grad_in, grad_out):

        self.gradients = grad_out[0]

    def forward(self, x):
        """ Forward pass with Grad-CAM support """
        outputs = self.CvT(x)


        feature_maps = outputs.last_hidden_state  
        #print(feature_maps)



        pooler_output = outputs.cls_token_value  
        pooler_output = pooler_output.squeeze(1)
        logits = self.classifier(pooler_output)  
        return logits

    def get_activations_gradient(self):

        return self.gradients

    def get_activations(self, x):

        _ = self.forward(x)  
        return self.feature_maps  



model = CvT().to(device)