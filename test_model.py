import torch
import torch.optim as optim
from yolo2 import Yolo, Yolo_Block
from loss import Loss
from utils2 import  AverageMeter,class_accuracy, use_gpu_if_possible
from dataset import get_data
import warnings
import time
import sys
import math
from torch.optim.optimizer import Optimizer
warnings.filterwarnings("ignore")
from train2 import test_model

if __name__ == "__main__":
    
    model = Yolo_Block(3,3,20).eval()
    model_dict=torch.load("model_RAdam_Augmented.pt", map_location = use_gpu_if_possible())
    model.load_state_dict(model_dict)
    
    S = [13,26]
    ANCHORS = [ [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]]
    
    train_loader, test_loader = get_data('train_pascal.csv','test_pascal.csv')
    
    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to("cuda:0")
    
    test_model(model, testloader, scaled_anchors, performance=class_accuracy)
    
    
    
    
    