import torch
import torch.nn as nn

from utils import intersection_over_union

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        

        
        self.lambda_class = 1
        self.lambda_noobj = 5
        self.lambda_obj = 10
        self.lambda_box = 1

    def forward(self, predictions, target, anchors):
        
        obj = target[..., 0] == 1  #  Iobj_i
        noobj = target[..., 0] == 0  #  Inoobj_i



        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

       

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )