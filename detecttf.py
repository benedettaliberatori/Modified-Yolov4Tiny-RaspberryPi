import cv2 as cv
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import torch
from torchvision import transforms
from utils2 import cells_to_bboxes, non_max_suppression

if __name__ == '__main__':
    print("Started")

    ANCHORS = [[(0.275,   0.320312), (0.068, 0.113281), (0.017,  0.03)],
               [(0.03,   0.056), (0.01,   0.018), (0.006,   0.01)]]
    S = [13, 26]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    interpreter = tf.lite.Interpreter('downblur.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details= interpreter.get_output_details()


    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("cannot open camera")
        exit()
    
    start = time.perf_counter()
    count = 0

    prev_frame_time = 0
    new_frame_time = 0

    history_fps = []

    while True:
        r, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (416, 416))

        font = cv.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()

        frame_tensor = transforms.ToTensor()(frame).unsqueeze_(0)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        interpreter.set_tensor(input_details[0]['index'], frame_tensor)
        interpreter.invoke()

        out = [0,0]
        out[1] = torch.from_numpy(interpreter.get_tensor(output_details[0]['index'])).float()
        out[0] = torch.from_numpy(interpreter.get_tensor(output_details[1]['index'])).float()
        
        
        boxes = []

        for i in range(2):
                anchor = scaled_anchors[i]
                boxes += cells_to_bboxes(out[i], S=out[i].shape[2], anchors = anchor)[0]
                
            
        boxes = non_max_suppression(boxes, iou_threshold= .3, threshold=.65, box_format = "midpoint")
        
        for box in boxes:

            if box[0] == 0: # mask
                    color = (0,250,154)
                    label = 'mask'
            else: # no mask
                    color = (255, 0, 0)
                    label = 'no mask'
            height, width = 416, 416
            p = box[1]
            box = box[2:]

            p0 = (int((box[0] - box[2]/2)*height) ,int((box[1] - box[3]/2)*width))
            p1 = (int((box[0] + box[2]/2)*height) ,int((box[1] + box[3]/2)*width))
    
            CV2_frame = cv.rectangle(frame, p0, p1, color, thickness=2)
            cv.putText(CV2_frame, label + "{:.2f}".format(p*100) + '%', (int((box[0] - box[2]/2)*height), int((box[1] - box[3]/2)*width)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            


        
        count += 1
        fps = 1/(new_frame_time-prev_frame_time)
        history_fps.append(fps)
        
        prev_frame_time = new_frame_time
        fps = "{:3.4f}".format(fps)
        fps = "FPS: " + fps
        cv.putText(frame, fps, (0, 30), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        
        

        cv.imshow('detecter', frame)
 
        c = cv.waitKey(1)
        if c == 27:
            cap.release()
            cv.destroyAllWindows()
            break
    
    plt.plot(history_fps)
    plt.ylabel("FPS")
    plt.show()
    end = time.perf_counter()
    print(count/(end-start))