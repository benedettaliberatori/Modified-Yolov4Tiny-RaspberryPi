import sys 
import numpy as np
import os
import pandas as pd
import torch
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.utils import iou_width_height 
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomYoloDataset(Dataset):
    """
    A custom Dataset class for the Yolo model
    """
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26],C=2, transform=None ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor( anchors[0] + anchors[1] )  
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 2
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # Conversion from  [class label, x, y, width, height] to [x, y, width, height, class label]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Data Augmentation 
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # 2 targets (from groundtruth) to be compared with the 2 outputs of the model
        targets = [torch.zeros((self.num_anchors // 2, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale,i, j, 0] = -1  

        return image, tuple(targets)

def get_data(train_csv_path, test_csv_path):
    """
    Gets train and test loader, performing Data Augmentation.
    Parameters: 
        train_csv_path & test_csv_path: paths to the csv containing the 
                                        files' names of images and labels.
    Returns: 
        train_loader & test_loader
    """

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
                A.ShiftScaleRotate( 
                    rotate_limit=20, p=0.5, border_mode=0
                ),
                A.Affine(shear=15, p=0.5),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255,),
        A.Downscale (scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=True, p=1),
        A.MotionBlur(p=1),

        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    
    )

    train_dataset = CustomYoloDataset(
        train_csv_path,
        transform=transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16],
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
    )
    test_dataset = CustomYoloDataset(
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