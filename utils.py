import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import YOLODataset




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

    

def get_data(train_csv_path, test_csv_path):
    

    IMAGE_SIZE = 416
    BATCH_SIZE = 16 
    NUM_WORKERS = 4
    DATASET = 'dataset'
    IMG_DIR = DATASET + "/images/"
    LABEL_DIR = DATASET + "/labels/"

    ANCHORS =  [[(0.275 ,   0.320312), (0.068    0.113281), (0.017  ,  0.03   )], 
              [(0.03     0.056   ), (0.01     0.018   ), (0.006    0.01    )]]

    
    transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],)

    train_dataset = YOLODataset(
        train_csv_path,
        transform=transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16],
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16],
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
 
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16],
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        
    )

    return train_loader, test_loader, train_eval_loader

    



        