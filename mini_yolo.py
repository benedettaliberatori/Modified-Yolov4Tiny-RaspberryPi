from utils2 import cells_to_bboxes, non_max_suppression, use_gpu_if_possible
from backbone import backbone
from CSP import ConvBlock
import torch.nn as nn
import torch
from torchvision import transforms
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
        #self.generate()

    def forward(self,x):
        out1 , out2 = self.back(x)
        out2 = self.conv1(out2)
        feat2 = out2
        #out2 = self.conv3(out2)
        #feat1 = torch.cat([out1,self.upsample(out2)],dim=1)
        feat2 = self.conv1(feat2)
        #feat1 = self.conv2(feat1)
        #feat1 = self.conv4(feat1)
        feat2 = self.conv5(feat2)
        return self.head(feat2).reshape(feat2.shape[0], self.B, 2 + 5, feat2.shape[2], feat2.shape[3]).permute(0, 1, 3, 4, 2)#, self.head(feat1).reshape(feat1.shape[0], self.B, 2 + 5, feat1.shape[2], feat1.shape[3]).permute(0, 1, 3, 4, 2)
    
class Yolo(object):

    def __init__(self,**kwargs):
        self.generate()
    
    def generate(self):
        self.net=Yolo_Block(3,3,2).eval()
        
        model_dict=torch.load("mini_yolo_untrained.pt", map_location = use_gpu_if_possible())
        self.net.load_state_dict(model_dict)
        

    def detect_Persson(self, CV2_frame,Tensor_frame, scaled_anchors, iou_thresh = .1, tresh = .65 ):
                       
        with torch.no_grad():

            out = self.net(Tensor_frame)
            boxes = []
            
            
            anchor = scaled_anchors[0]
                #print(anchor.shape)
                #print(out[i].shape)
            
            boxes += cells_to_bboxes(out, S=out.shape[2], anchors = anchor)[0]
                
            boxes = non_max_suppression(boxes, iou_threshold= iou_thresh, threshold=tresh, box_format = "midpoint")
            

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
                
                #print(p0)
                #print(p1)
                
                CV2_frame = cv2.rectangle(CV2_frame, p0, p1, color, thickness=2)
                cv2.putText(CV2_frame, label + "{:.2f}".format(p*100) + '%', (int((box[0] - box[2]/2)*height), int((box[1] - box[3]/2)*width)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            return CV2_frame          
            

if __name__ == '__main__':
     
    model = Yolo_Block(3, 3, 2)
    torch.save(model.state_dict(), 'mini_yolo_untrained.pt')