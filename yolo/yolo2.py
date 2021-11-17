import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.utils import cells_to_bboxes, non_max_suppression, use_gpu_if_possible
from yolo.backbone import backbone
from yolo.CSP import ConvBlock
import torch.nn as nn
import torch
import cv2





class Yolo_Block(nn.Module):
    
    def __init__(self,in_channels,B,num_classes):
        super().__init__()
        self.back = backbone(in_channels)
        self.conv1 = ConvBlock(512,512,3,1)
        self.conv2 = ConvBlock(512,256,3,1)
        self.conv3 = nn.Conv2d(512,128,1,1)
        self.upsample = nn.ConvTranspose2d(128,256,2,2)
        self.conv4 = nn.Conv2d(256,255,1,1)
        self.conv5 = nn.Conv2d(512,255,1,1)
        self.head = nn.Conv2d(255,B*(5+num_classes),1,1)
        self.B = B
        

    def forward(self,x):
        out1 , out2 = self.back(x)
        out2 = self.conv1(out2)
        feat2 = out2
        out2 = self.conv3(out2)
        feat1 = torch.cat([out1,self.upsample(out2)],dim=1)
        feat2 = self.conv1(feat2)
        feat1 = self.conv2(feat1)
        feat1 = self.conv4(feat1)
        feat2 = self.conv5(feat2)
        return self.head(feat2).reshape(feat2.shape[0], self.B, 2 + 5, feat2.shape[2], feat2.shape[3]).permute(0, 1, 3, 4, 2),self.head(feat1).reshape(feat1.shape[0], self.B, 2 + 5, feat1.shape[2], feat1.shape[3]).permute(0, 1, 3, 4, 2)
    
class Yolo(object):

    def __init__(self,**kwargs):
        self.generate()
    
    def generate(self):
        """
        Loads a pre-trained pytorch model and
        sets it to evaluation mode. 
        """
        self.net=Yolo_Block(3,3,2).eval()
        model_dict=torch.load("../models/model.pt", map_location = use_gpu_if_possible())
        self.net.load_state_dict(model_dict)
        

    def detect(self, CV2_frame,Tensor_frame, scaled_anchors, iou_thresh = .1, tresh = .65 ):
        """
        Used to get predictions from camera stream. 
        Returns a 416x416 image in RGB.

        INPUTS:
            CV2_frame, RGB 416x416 image read from camera 
            Tensor_frame, image in tensor format
            scaled_anchors, anchor boxes rescaled 
            iou_thresh =  Intersection Over Union threshold
            tresh = threshold to filter boxes w/ smaller objectness score
        

        """
                       
        with torch.no_grad():

            out = self.net(Tensor_frame)

            boxes = []
            
            for i in range(2):
                anchor = scaled_anchors[i]
                boxes += cells_to_bboxes(out[i], S=out[i].shape[2], anchors = anchor)[0]
                
            boxes = non_max_suppression(boxes, iou_threshold= iou_thresh, threshold=tresh)
            

            for box in boxes:
                if box[0] == 0: 
                        #color = (0,250,154)
                        color = (0, 100, 0)
                        label = 'MASK'
                else: 
                        #color = (255, 0, 0)
                        color = (139, 0,0)
                        label = 'NO MASK'
                height, width = 416, 416

                p = box[1]
                box = box[2:]
                p0 = (int((box[0] - box[2]/2)*height) ,int((box[1] - box[3]/2)*width))
                p1 = (int((box[0] + box[2]/2)*height) ,int((box[1] + box[3]/2)*width))
                
                
                
                CV2_frame = cv2.rectangle(CV2_frame, p0, p1, color, thickness=2)
                #cv2.putText(CV2_frame, label + "{:.2f}".format(p*100) + '%', (int((box[0] - box[2]/2)*height), 
                            #int((box[1] - box[3]/2)*width)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                #cv2.putText(CV2_frame, label , (int((box[0] - box[2]/2)*height), 
                #            int((box[1] - box[3]/2)*width)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
                p1 = p0[0] + w, p0[1] + h + 3

                cv2.rectangle(CV2_frame, p0, p1, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(CV2_frame, label, ( p0[0], p0[1] + h + 2), 0, 2 / 3, (255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
            return CV2_frame           




