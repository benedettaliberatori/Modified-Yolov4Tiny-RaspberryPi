import torch
import torch.nn as nn

from utilities.utils import intersection_over_union

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
        """
        INPUT:
            predictions: output of the model at one scale (torch.Size([3, 13, 13, 6]) or torch.Size([3, 26, 26, 6]))
            target: ground truth at the same scale (same size of predictions)
            anchors: anchor boxes

        
        """
        # Boolean tensors: 
        obj = target[..., 0] == 1  #  Iobj_i : True iff a cell has objects 
                                   #  assigned w/ that anchor box
        noobj = target[..., 0] == 0  #  Inoobj_i : True iff a cell 
                                     #  has no objects assigned w/ that anchor box


        # Binary Cross Entropy w/ Logits Loss for No Object Loss
        # computed using the objectness score of the anchors in cells 
        # with no object assigned
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # Compute Intersection Over Union between 
        # target bounding boxes and the predicted 
        # bounding boxes in the output
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        
        # Binary Cross Entropy w/ Logits Loss for Object Loss
        # computed as the previuos one but for the anchors 
        # that have an object assigned & also
        # multiplying targets for the iou
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        
        # Sigmoid function to (x,y) 
        # to ensure they are between [0,1]
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  

        # Widths and heights 
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        ) 
        # Mean Squared Error Loss for Bounding Boxes Loss
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

       
        # Cross Entropy loss for Classification Loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),)

        

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )