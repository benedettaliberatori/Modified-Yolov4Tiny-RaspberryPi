import sys 
sys.path.append("..")

import cv2 as cv
from yolo.yolo2 import Yolo
import torch
from torchvision import transforms
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ANCHORS = [[(0.275,   0.320312), (0.068, 0.113281), (0.017,  0.03)],
               [(0.03,   0.056), (0.01,   0.018), (0.006,   0.01)]]
    S = [13, 26]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    model = Yolo()

    

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
        
         
        frame = model.detect_Persson(frame, frame_tensor, scaled_anchors)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
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
