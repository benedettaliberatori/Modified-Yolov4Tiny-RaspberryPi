"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils2 import iou_width_height 

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26],
        C=2,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(
            anchors[0] + anchors[1])  # for all 2 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 2
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(
            self.label_dir, self.annotations.iloc[index, 1])
        bboxes2 = np.roll(np.loadtxt(fname=label_path,
                                    delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = []
        for box in bboxes2:
            bboxes.append(box[:4])



        # apply augmentations with albumentations
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        i = 0
        for box in bboxes:
            bboxes2[i][:4] = box
            i = i+1

        # Building the targets below:
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 2, S, S, 6))
                   for S in self.S]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)

def get_data(train_csv_path, test_csv_path):
    

    IMAGE_SIZE = 416
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DATASET = 'dataset'
    IMG_DIR = DATASET + "/images/"
    LABEL_DIR = DATASET + "/labels/"

    ANCHORS = [[(0.289062, 0.339265), (0.02 ,  0.035), (0.007 ,   0.012   )], [(0.035 , 0.064  ), (0.012 ,0.021), (0.08  , 0.129  )]]

    
    transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0
        ),
        A.OneOf(
            [
                A.ShiftScaleRotate( # Randomly apply affine transforms: translate, scale and rotate the input.
                    rotate_limit=20, p=0.5, border_mode=0
                )
                #A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    
    )

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
        pin_memory=True,
        
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
 
    )

    

    return train_loader, test_loader

    

if __name__ == '__main__':

    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
    data = YOLODataset('target.csv', 'images', 'labels', ANCHORS)

    print(data[3])
