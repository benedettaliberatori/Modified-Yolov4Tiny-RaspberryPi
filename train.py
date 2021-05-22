from re import M
import torch
import torch.optim as optim
from yolo import Yolo
from loss import Loss
from tqdm import tqdm
from utils import  AverageMeter,class_accuracy
from dataset import get_data
import warnings
warnings.filterwarnings("ignore")

def train_epoch(train_loader, model, optimizer, loss_fn, scaled_anchors,device,loss_meter, performance_meter, performance):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for x, y in train_loader:
        print(x.shape)
        x = x.to(device)
        y0, y1= (
            y[0].to(device),
            y[1].to(device),
        )


        out = model(x)
        print(out[0].shape)
        loss = (
            loss_fn(out[0], y0, scaled_anchors[0])
            + loss_fn(out[1], y1, scaled_anchors[1])
        )

        losses.append(loss.item())
        optimizer.zero_grad()

        acc = performance(model, train_loader,device)
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        performance_meter.update(val=acc, n=X.shape[0])
      



def train_model(train_loader, model, optimizer, loss_fn, scaled_anchors,device, performance):
    # create the folder for the checkpoints (if it's not None)

    
    model = model.to(device)
    model.train()

    # epoch loop
    for epoch in range(num_epochs):

        loss_meter = AverageMeter()
        performance_meter = AverageMeter()

        # added print for LR
        print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        


        train_epoch(train_loader, model, optimizer, loss_fn, scaled_anchors,device,loss_meter, performance_meter, performance)

        print(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}; Performance: {performance_meter.avg:.4f}")

        

    return loss_meter.sum, performance_meter.avg




if __name__ == "__main__":

    num_anchor = 6
    model = Yolo(3,num_anchor,2)
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.03
    )
    loss_fn = Loss()
    S=[13, 26]
    num_epochs = 1000

    ANCHORS =  [[(0.275 ,   0.320312), (0.068   , 0.113281), (0.017  ,  0.03   )], 
              [(0.03  ,   0.056   ), (0.01  ,   0.018   ), (0.006 ,   0.01    )]]

    train_loader, test_loader = get_data('train.csv','test.csv')

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to("cpu")

    train_model(train_loader, model, optimizer, loss_fn, scaled_anchors,None, performance=class_accuracy)

    