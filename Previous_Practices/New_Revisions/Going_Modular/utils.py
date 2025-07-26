


from pathlib import Path
import torch

def save_model(model_name:str , model:torch.nn.Module , target_dir:str):
    MODEL_PATH =Path(target_dir).resolve()
    MODEL_PATH.mkdir(parents=True,exist_ok=True)

    MODEL_NAME = model_name
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME 

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)