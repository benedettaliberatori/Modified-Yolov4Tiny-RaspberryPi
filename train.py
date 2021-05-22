from re import M
import torch
import torch.optim as optim
from yolo import Yolo
from loss import Loss
from tqdm import tqdm
from utils import get_data

import warnings
warnings.filterwarnings("ignore")

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(cpu)
        y0, y1, y2 = (
            y[0].to(cpu),
            y[1].to(cpu),
            y[2].to(cpu),
        )


        out = model(x)
        loss = (
            loss_fn(out[0], y0, scaled_anchors[0])
            + loss_fn(out[1], y1, scaled_anchors[1])
            + loss_fn(out[2], y2, scaled_anchors[2])
        )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)




if __name__ == "__main__":

    num_anchor = 6
    model = Yolo(3,num_anchor,2)
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.03
    )
    loss_fn = Loss()

    num_epochs = 1000

    ANCHORS =  [[(0.275 ,   0.320312), (0.068    0.113281), (0.017  ,  0.03   )], 
              [(0.03     0.056   ), (0.01     0.018   ), (0.006    0.01    )]]

    train_loader, test_loader, train_eval_loader = get_data('target.csv','target.csv')

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(cpu)

    
    for epoch in range(num_epochs):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)