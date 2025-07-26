from typing import List,Tuple
from PIL import Image
import torch
from torchvision import transforms
import torchvision 
import matplotlib.pyplot as plt




device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image (model: torch.nn.Module , image_path:str , class_names:List[str] , image_size:Tuple[int,int]=(224,224) , transform:torchvision.transforms=None , device: torch.device = device):
    
    
    img = Image.open(image_path)
    
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([ transforms.Resize(image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485,0.456,0.406] , std=[0.229,0.224,0.225]) ])

    
    model = model.to(device)
    
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0).to(device)
        
        target_image_pred = model(transformed_image)
        
    probabilites = torch.softmax(target_image_pred , dim=1)
    
    label = torch.argmax(probabilites,dim=1)
    
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[label]}  | Probs: {probabilites.max():.3f} ")
    plt.axis(False)
    plt.show()
        
        
        
        