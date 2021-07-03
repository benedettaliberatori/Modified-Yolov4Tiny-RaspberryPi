import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from yolo.yolo2 import Yolo_Block
from utilities.utils import  mean_average_precision,get_evaluation_bboxes, use_gpu_if_possible
from dataset.dataset import get_data
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]]
    


    train_loader, test_loader = get_data('dataset/train.csv','dataset/test.csv')

    model = Yolo_Block(3,3,2).eval()
    model_dict=torch.load("models/downblur.pt", map_location = use_gpu_if_possible())
    model.load_state_dict(model_dict)

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold = 0.5,
        anchors=ANCHORS,
        threshold=0.05
    )

    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        num_classes=2,
    )

    print(f"MAP: {mapval.item()}")