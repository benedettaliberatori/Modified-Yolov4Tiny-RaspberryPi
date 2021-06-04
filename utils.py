
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from torchvision.ops import nms

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


def NMS(prediction, conf_thres= 0.8, nms_thres = 0.25):
    box_temp=prediction.new(prediction.shape)
    box_temp[:,:,0]=prediction[:,:,0]-prediction[:,:,2]/2
    box_temp[:,:,1]=prediction[:,:,1]-prediction[:,:,3]/2
    box_temp[:,:,2]=prediction[:,:,0]+prediction[:,:,2]/2
    box_temp[:,:,3]=prediction[:,:,1]+prediction[:,:,3]/2
    prediction[:,:,:4]=box_temp[:,:,:4]
    
    output = [None for _ in range(len(prediction))]
    
    for index, pred in enumerate(prediction):
        cls_conf, cls_pred = torch.max(pred[:,5:7], dim=1, keepdim=True)
        score = pred[:,4]*cls_conf[:,0]
        pred=pred[score>conf_thres] # squeeze?
        
        cls_conf=cls_conf[(score>conf_thres)]
        cls_pred=cls_pred[(score>conf_thres)]
        
        if pred.size(0) == 0:
            continue
        
        detection=torch.cat((pred[:,:5],cls_conf.float(),cls_pred.float()),dim=1) # 6 values (coordinates, class_conf and cls_pred)
        
        for cls in range(2):
            is_cls = detection[:,-1]==cls
            
            detected_class = detection[is_cls]
            
            boxes = detected_class[:,:4]
            score = detected_class[:,4]*detected_class[:,5]
            
            keep = nms(boxes, score, nms_thres)
            
            max_detection = detected_class[keep]
            
            output[index] = max_detection if output[index] is None else torch.cat((output[index], max_detection))
            
    return output      



def iou_width_height(boxes1, boxes2):
    
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels):
    


    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  # x.clamp.(0) max(0, x)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
def class_accuracy(out, model, y, device, threshold=0.5):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0


    #x = x.to(device)
    #with torch.no_grad():
    #    out = model(x)
    for i in range(2):
        y[i] = y[i].to(device)
        obj = y[i][..., 0] == 1 # in paper this is Iobj_i
        noobj = y[i][..., 0] == 0  # in paper this is Iobj_i
        correct_class += torch.sum(
            torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
        )
        tot_class_preds += torch.sum(obj)
        obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
        correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
        tot_obj += torch.sum(obj)
        correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
        tot_noobj += torch.sum(noobj)

    #print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    #print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    #print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()

    return (correct_class/(tot_class_preds+1e-16))*100 ,(correct_obj/(tot_obj+1e-16))*100 , (correct_noobj/(tot_noobj+1e-16))*100





def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"
        