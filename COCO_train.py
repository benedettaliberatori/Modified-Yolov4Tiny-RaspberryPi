import torch
import torch.optim as optim
from yolo import Yolo
from loss import Loss
from utils2 import  AverageMeter,class_accuracy, use_gpu_if_possible
from dataset import get_data
import warnings
import time
import sys
import math
from torch.optim.optimizer import Optimizer
from train2 import RAdam, train_epoch, train_model, test_model
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    num_anchor = 6
    model = Yolo(3, num_anchor //2, 80)

    torch.save(model.state_dict(), 'untrained.pt') 
    
    optimizer_SGD = optim.SGD(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )
    optimizer_RAdam = RAdam(model.parameters(), lr=0.001/5, weight_decay=0.005)
    loss_fn = Loss()
    S=[13, 26]
    num_epochs = 100
#
    ANCHORS = [[(0.276, 0.320312), (0.068,  0.113281), (0.03, 0.056 )], 
               [(0.017, 0.03), (0.01, 0.018), (0.006, 0.01)]]
#
    train_loader, test_loader = get_data('train_coco.csv','test_coco.csv')
#
    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to("cuda:0")
#   

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_SGD, step_size=20, gamma=1.1)
    if str(sys.argv[-1]) == "SGD":
        optimizer = optimizer_SGD
        model_save_name = 'model_SGD.pt'
        scheduler = scheduler
    
    if str(sys.argv[-1]) == "RADAM":
        optimizer = optimizer_RAdam
        model_save_name = 'COCO_RAdam.pt'
        scheduler = None

    scaler = torch.cuda.amp.GradScaler()
    train_model(train_loader, model, optimizer, loss_fn, num_epochs, scaler,  scaled_anchors,None, performance=class_accuracy,lr_scheduler=scheduler,epoch_start_scheduler= 40)
    
    
    path = F"{model_save_name}" 
    torch.save(model.state_dict(), path)