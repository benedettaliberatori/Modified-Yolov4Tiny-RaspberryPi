import cv2 as cv
from my_quantized import Yolo_Q
import torch
from torchvision import transforms
import time

if __name__ == "__main__":

    ANCHORS = [[(0.275,   0.320312), (0.068, 0.113281), (0.017,  0.03)],
               [(0.03,   0.056), (0.01,   0.018), (0.006,   0.01)]]
    S = [13, 26]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    model = Yolo_Q()

    print("here")

    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("cannot open camera")
        exit()
    
    start = time.perf_counter()
    count = 0
    while True:
        r, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (416, 416))
        frame_tensor = transforms.ToTensor()(frame).unsqueeze_(0)
        
         
        frame = model.detect_Persson(frame, frame_tensor, scaled_anchors)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        count += 1
        cv.imshow('detecter', frame)
 
        c = cv.waitKey(1)
        if c == 27:
            cap.release()
            cv.destroyAllWindows()
            break
    
    end = time.perf_counter()
    print(count/(end-start))
