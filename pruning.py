import torch
import torch.optim as optim
from yolo import Yolo
from loss import Loss
from utils2 import  AverageMeter,class_accuracy, use_gpu_if_possible
from dataset import get_data
import warnings
import sys
warnings.filterwarnings("ignore")
from train2 import RAdam, train_model

def magnitude_pruning(net, p, mask=None, params_to_prune=[]):
    flat = []

    for i, (name, par) in enumerate(net.named_parameters()):
        if any([l in name for l in params_to_prune]):
            if mask is None:
                flat.append(par.abs().flatten())
            else:
                flat.append(par[mask[i]!=0].abs().flatten())
    flat = torch.cat(flat, dim=0).sort()[0]

    position = int(p * flat.shape[0])
    thresh = flat[position]

    new_mask = []
    for name, par in net.named_parameters():
        if any([l in name for l in params_to_prune]):
            m = torch.where(par.abs() >= thresh, 1, 0)
            new_mask.append(m)
        else:
            new_mask.append(torch.ones_like(par))
    
    return new_mask


def apply_mask(net, mask):
    print("Applied mask\n")
    for p, m in zip(net.parameters(), mask):
        p.data *= m

def pct_of_ones_in_mask(mask):
    return sum([m.sum().item() for m in mask]) / sum([m.numel() for m in mask])


if __name__ == "__main__":

            
    train_loader, test_loader = get_data('train.csv','test.csv')
    
    
    num_anchor = 6
    model = Yolo(3, num_anchor //2, 2)
    model.load_state_dict(torch.load('model_RAdam.pt'))
    S=[13, 26]
    
    ANCHORS = [[(0.276  , 0.320312), (0.068  ,  0.113281), (0.03   ,  0.056    )], [(0.017 ,   0.03  ), (0.01 ,  0.018  ), (0.006  , 0.01 )]]

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to("cuda:0")

  
    optimizer = RAdam(model.parameters(), lr=0.001/5, weight_decay=0.005)
    loss_fn = Loss()
    scaler = torch.cuda.amp.GradScaler()
    num_epochs = int(sys.argv[-1])

    print("Step 1")
    mask = magnitude_pruning(model, .2, params_to_prune=["0", "1"])
    print(pct_of_ones_in_mask(mask))
    apply_mask(model, mask)
    train_model(train_loader, model, optimizer, loss_fn, num_epochs, scaler, scaled_anchors, None, performance=class_accuracy, epoch_start_scheduler=1)

    for i in range(5):
        
        print(f"Step {i+1}")
        model.to("cpu")        
        mask = magnitude_pruning(model, .2, params_to_prune=["0", "1"], mask=mask)
        print(pct_of_ones_in_mask(mask))
        apply_mask(model, mask)
        train_model(train_loader, model, optimizer, loss_fn, num_epochs , scaler, scaled_anchors, None, performance=class_accuracy, epoch_start_scheduler=1)



    torch.save(model.state_dict(), f"pruned_RAdam_{num_epochs}.pt")

    model_LTH = Yolo(3,3,2)
    model_LTH.load_state_dict(torch.load("untrained.pt"))

    apply_mask(model_LTH, mask)

    torch.save(model_LTH.state_dict(), f"LTH.pt")


