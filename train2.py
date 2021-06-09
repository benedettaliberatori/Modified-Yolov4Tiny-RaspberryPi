import torch
import torch.optim as optim
from yolo import Yolo
from loss import Loss
from utils2 import  AverageMeter,class_accuracy, use_gpu_if_possible
from dataset import get_data
import warnings
import time
import sys
warnings.filterwarnings("ignore")


def train_epoch(train_loader, model, optimizer, loss_fn, scaler,  scaled_anchors,device,loss_meter,performance_meter_class,performance_meter_obj,performance_meter_noobj, performance):


    for x, y in train_loader:

        x = x.to(device)
        y0, y1= (
            y[0].to(device),
            y[1].to(device),
        )

        
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        with torch.cuda.amp.autocast():
            out = model(x)
    
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
            )
        
       
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        acc1 , acc2 , acc3 = performance(out,model, y,device)
        
        loss_meter.update(val=loss.item(), n=x.shape[0])


        performance_meter_class.update(val=acc1, n=x.shape[0])
        performance_meter_obj.update(val=acc2, n=x.shape[0])
        performance_meter_noobj.update(val=acc3, n=x.shape[0])

      



def train_model(train_loader, model, optimizer, loss_fn, num_epochs, scaler, scaled_anchors,device, performance, lr_scheduler=None, epoch_start_scheduler=1):
    
    torch.backends.cudnn.benchmark = True
    if device is None:
        device = use_gpu_if_possible()
    
    model = model.to(device)
    model.train()

    # epoch loop
    for epoch in range(num_epochs):

        loss_meter = AverageMeter()
        performance_meter_class = AverageMeter()
        performance_meter_obj = AverageMeter()
        performance_meter_noobj = AverageMeter()

        # added print for LR
        print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        

        start = time.perf_counter()
        train_epoch(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors,device,loss_meter, performance_meter_class,performance_meter_obj,performance_meter_noobj, performance)
        end = time.perf_counter()

        print(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}; Performance_class: {performance_meter_class.avg:.4f};")
        print(f"Performance_obj: {performance_meter_obj.avg:.4f}; Performance_noobj: {performance_meter_noobj.avg:.4f};")
        print(f"Elapsed time: {end-start:.4f};")

        if lr_scheduler is not None:
            if epoch >= epoch_start_scheduler:
                lr_scheduler.step()

    return loss_meter.sum, performance_meter_class.avg, performance_meter_obj.avg, performance_meter_noobj.avg


def test_model(model, dataloader,scaled_anchors, performance=class_accuracy, loss_fn=None, device=None):
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = AverageMeter()
    
    if device is None:
        device = use_gpu_if_possible()

    model = model.to(device)

    performance_meter_class = AverageMeter()
    performance_meter_obj = AverageMeter()
    performance_meter_noobj = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            #y = y.to(device)
            y0, y1= (
            y[0].to(device),
            y[1].to(device),
        )
            out = model(X)
            loss = (
            loss_fn(out[0], y0, scaled_anchors[0])
            + loss_fn(out[1], y1, scaled_anchors[1])
        )
        
            acc1 , acc2 , acc3  =  performance(out,model, y,device)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter_class.update(val=acc1, n=X.shape[0])
            performance_meter_obj.update(val=acc2, n=X.shape[0])
            performance_meter_noobj.update(val=acc3, n=X.shape[0])
    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf_class, fin_perf_obj , fin_perf_noobj = performance_meter_class.avg , performance_meter_obj.avg , performance_meter_noobj.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance_class {fin_perf_class:.4f} , performance_obj {fin_perf_obj:.4f} ,performance_noobj {fin_perf_noobj:.4f}")
    return fin_loss, fin_perf_class, fin_perf_obj , fin_perf_noobj 

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)
    
def load_checkpoint(model, optmizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optmizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch


if __name__ == "__main__":

    num_anchor = 6
    model = Yolo(3, num_anchor //2, 2)
    optimizer_SGD = optim.SGD(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )
    optimizer_RMS = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_fn = Loss()
    S=[13, 26]
    num_epochs = 100
#
    ANCHORS = [[(0.276  , 0.320312), (0.068  ,  0.113281), (0.03   ,  0.056    )], [(0.017 ,   0.03  ), (0.01 ,  0.018  ), (0.006  , 0.01 )]]
#
    train_loader, test_loader = get_data('train.csv','test.csv')
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
    
    if str(sys.argv[-1]) == "SGD2":
        optimizer = optimizer_SGD
        model_save_name = 'model_SGD_no_scheduler_200.pt'
        scheduler = None

    scaler = torch.cuda.amp.GradScaler()
    train_model(train_loader, model, optimizer, loss_fn, num_epochs, scaler,  scaled_anchors,None, performance=class_accuracy,lr_scheduler=scheduler,epoch_start_scheduler= 40)
    
    
    path = F"{model_save_name}" 
    torch.save(model.state_dict(), path)
    
    #model = Yolo(3, 6//2, 2)
    #model.load_state_dict(torch.load('model.pt'))
    #test_model(model, test_loader, scaled_anchors, performance=class_accuracy, loss_fn= Loss(), device='cuda:0')
    