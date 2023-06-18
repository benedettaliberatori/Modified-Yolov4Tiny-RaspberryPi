import sys 
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import time
import tensorflow as tf
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from utilities.utils import cells_to_bboxes, non_max_suppression
import pandas as pd

DESIRED_SHAPE = 416
PAD = True

DATASET_NAME_TO_PATH = {
    "test_set_select" : "dataset_extra/test_set_select",
    "test_set_batchII" : "dataset_extra/test_set_batchII",
    "fairface_train" : "dataset_extra/fairface_dataset/train",
    "fairface_test" : "dataset_extra/fairface_dataset/test",
}

def resize(frame):
    old_size = frame.shape[:2] # old_size is in (height, width) format

    ratio = float(DESIRED_SHAPE)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    frame = cv2.resize(frame, (new_size[1], new_size[0]))

    delta_w = DESIRED_SHAPE - new_size[1]
    delta_h = DESIRED_SHAPE - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return frame


def eval_dataset(dataset_name):
    dataset_path = DATASET_NAME_TO_PATH[dataset_name]
    print(dataset_path)

    j = 0
    N = []
    Y = []
    X = []
    W = []
    H = []
    P = []
    L = []
    
    for f in tqdm(os.listdir(dataset_path)):

        img_path = os.path.join(dataset_path, f)
        #print(img_path)

        image = cv2.imread(img_path) 

        original_height = image.shape[0]
        original_width = image.shape[1]

        #print((original_height, original_width))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if PAD:
            image = resize(image)
        else:
            image = cv2.resize(image, (416, 416))

        frame_tensor = transforms.ToTensor()(image).unsqueeze_(0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        interpreter.set_tensor(input_details[0]['index'], frame_tensor)
        interpreter.invoke()

        out = [0,0]
        out[1] = torch.from_numpy(interpreter.get_tensor(output_details[0]['index'])).float()
        out[0] = torch.from_numpy(interpreter.get_tensor(output_details[1]['index'])).float()
        
        
        boxes = []

        for i in range(2):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(out[i], S=out[i].shape[2], anchors = anchor)[0]
                
            
        boxes = non_max_suppression(boxes, iou_threshold= .1, threshold=.65)

        #print(f"Found {len(boxes)} boxes")

        if len(boxes) == 0:
            N.append(f)
            X.append(None)
            Y.append(None)
            W.append(None)
            H.append(None)
            P.append(None)
            L.append(None)
        
        for box in boxes:

            if box[0] == 0: # mask
                    color = (0,250,154)
                    label = 'mask'
            else: # no mask
                    color = (255, 0, 0)
                    label = 'no mask'

            height, width = original_height, original_width
            height, width = 416, 416

            p = box[1]
            box = box[2:]

            #print(f"Y = {box[0] * original_height}")
            #print(f"X = {box[1] * original_width}")

            #print(f"H = {box[2] * original_height}")
            #print(f"W = {box[3] * original_width}")

            #print(f"Confidence = {p*100}")

            N.append(f)
            X.append(box[0])
            Y.append(box[1])
            H.append(box[2])
            W.append(box[3])
            P.append(p * 100)
            L.append(label)

            #p0 = (int((box[0] - box[2]/2)*height) ,int((box[1] - box[3]/2)*width))
            #p1 = (int((box[0] + box[2]/2)*height) ,int((box[1] + box[3]/2)*width))
    
            #image = cv2.resize(image, (width, height))

            #CV2_frame = cv2.rectangle(image, p0, p1, color, thickness=2)
            #cv2.putText(CV2_frame, label + "{:.2f}".format(p*100) + '%', (int((box[0] - box[2]/2)*height), int((box[1] - box[3]/2)*width)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            #cv2.imshow('detecter', image)

    df = pd.DataFrame({
        "N" : N,
        "X" : X,
        "Y" : Y,
        "W" : W,
        "H" : H,
        "P" : P,
        "L" : L
        }
    )

    results_file_name = "dataset_extra/results/" + dataset_name
    if PAD: results_file_name = results_file_name + "_padded"
    results_file_name = results_file_name + ".csv"
    df.to_csv(results_file_name, index=False)

if __name__ == '__main__':
    

    ANCHORS = [[(0.275,   0.320312), (0.068, 0.113281), (0.017,  0.03)],
               [(0.03,   0.056), (0.01,   0.018), (0.006,   0.01)]]
    S = [13, 26]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    
    interpreter = tf.lite.Interpreter('models/pmodel.tflite')

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details= interpreter.get_output_details()

    for dataset_name in DATASET_NAME_TO_PATH.keys():
        eval_dataset(dataset_name)
    