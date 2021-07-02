import sys 
sys.path.append("..")

from yolo.backbone import backbone
from yolo.CSP import ConvBlock

import torch.nn as nn
import torch




class Yolo(nn.Module):
    
           
    def __init__(self,in_channels,B,num_classes):
        torch.manual_seed(1)
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
        self.num_classes = num_classes

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
        return self.head(feat2).reshape(feat2.shape[0], self.B, self.num_classes + 5, feat2.shape[2], feat2.shape[3]).permute(0, 1, 3, 4, 2),self.head(feat1).reshape(feat1.shape[0], self.B, self.num_classes + 5, feat1.shape[2], feat1.shape[3]).permute(0, 1, 3, 4, 2)




if __name__ == '__main__':
    
    x = torch.rand(1,3,416,416)
    model = Yolo(3,3,2)
    model.detect(x)
    #odel = Yolo(3,20,5)
    #out1,out2 = model(x)

    #print('out1 :',out1.shape,'out2:',out2.shape)


        

