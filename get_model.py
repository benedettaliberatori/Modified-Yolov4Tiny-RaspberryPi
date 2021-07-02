from yolo2 import Yolo, Yolo_Block
import torch

if __name__ == "__main__":
    
    model=Yolo_Block(3,3,2).train()
        
    model_dict=torch.load("model_RAdam_Augmented.pt", map_location = 'cpu')
    model.load_state_dict(model_dict)
    
    torch.save(model, "torch_model.pt")
    
    model_path = "torch_model.pt"
    print(model_path)
    loaded_model = torch.load(model_path, map_location='cpu')
    loaded_model.eval()
    