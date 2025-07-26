import os
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

CORES = os.cpu_count()





def create_dataloaders(train_dir:str , test_dir:str , train_transforms:transforms.Compose , test_transforms:transforms.Compose , batch_size:int , num_worker: int = CORES):

    """ Creates training and testing dataloaders
    
    
    Args:

        train_dir: Path to training directory.
        test_dir: Path to test directory.
        train_transforms = transformations to be applied to the training set.
        test_transforms = transformations to be applied on the test set.
        batch_size = Number of samples per batch in the dataloaders. 
        num_workers = An integer for the number of workers per DataLoader.
    
    
    
    Returns:
        A tuple of (train_dataloader, test_dataloader , class_names)
        Where class_names is a list of the target classes.
        
        Example usage:
            train_dataloader , test_dataloader, class_names = create_dataloader(train_dir = path/to/train_dir , test_dir = path/to/test_dir , train_transforms = some_transforms , test_transforms = some_transforms , batch_size = 32 , num_workers = 4 )
             
    """



    train_data = datasets.ImageFolder(root = train_dir,transform = train_transforms)
    test_data = datasets.ImageFolder(root = test_dir, transform = test_transforms)
    class_names = train_data.classes
    train_dataloader = DataLoader( dataset= train_data , batch_size = batch_size , shuffle=True , num_workers=num_worker , pin_memory=True)
    test_dataloader = DataLoader( dataset= test_data , batch_size = batch_size , num_workers=num_worker , shuffle=False , pin_memory=True)





    return train_dataloader , test_dataloader , class_names




