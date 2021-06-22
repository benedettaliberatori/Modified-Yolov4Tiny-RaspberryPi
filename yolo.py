from backbone import backbone
from CSP import ConvBlock
from PIL import ImageDraw
import torch.nn as nn
import torch
import numpy as np
from utils2 import use_gpu_if_possible


class DecodeBox(nn.Module):

    def __init__(self, scaled_anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.scaled_anchors = scaled_anchors
        self.num_anchors = len(scaled_anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self,input):

        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        prediction = input

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        anchor_w = FloatTensor(self.scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(self.scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        pred_boxes = FloatTensor(prediction[..., :4].shape)

        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.reshape(batch_size, -1, 4)*scale,
                            conf.reshape(batch_size, -1, 1), pred_cls.reshape(batch_size, -1, self.num_classes)), -1)
        return output.data





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

    def detect(self,frame):
        
        confidence = 0.9
        
        ANCHORS =  [[(0.275 ,   0.320312), (0.068   , 0.113281), (0.017  ,  0.03   )], 
           [(0.03  ,   0.056   ), (0.01  ,   0.018   ), (0.006 ,   0.01    )]]
        
        self.feat_decoder=[]
        
        self.net=Yolo(3, 6//2, 2).eval()
        model_dict=torch.load("model_100_epochs.pt", map_location = use_gpu_if_possible())
        self.net.load_state_dict(model_dict)


        for i in range(2):
            decoder=DecodeBox(ANCHORS[i],2,(416,416))
            self.feat_decoder.append(decoder)

        #img = torch.from_numpy(frame.array)
        img = frame
        out_list=[]

        with torch.no_grad():

            out_13, out_26 = self.net(img)

            decoder_out=self.feat_decoder[0](out_13)
            out_list.append(decoder_out)
            decoder_out=self.feat_decoder[1](out_26)
            out_list.append(decoder_out)

            output=torch.cat(out_list,dim=1)
            
            # print(len(output))
            # print(output.shape)
            
            batch_detection = NMS(output)
            
            # print(len(batch_detection))
            # print(batch_detection)

            end_score = batch_detection[:,4] * batch_detection[:,5]
            end_index = end_score > confidence
            
            end_label=np.array(batch_detection[end_index,-1],np.int32)
            end_boxes=np.array(batch_detection[end_index,:4])
            # end_xmin = np.expand_dims(end_boxes[:, 0], -1)
            # end_ymin = np.expand_dims(end_boxes[:, 1], -1)
            # end_xmax = np.expand_dims(end_boxes[:, 2], -1)
            # end_ymax = np.expand_dims(end_boxes[:, 3], -1)
            
            for i, c in enumerate(end_label):
                # score=end_score[i]
                top,left,bottom,right=end_boxes[i]
                
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottem = min(np.shape(frame)[0], np.floor(bottem + 0.5).astype('int32'))
                right = min(np.shape(frame)[1], np.floor(right + 0.5).astype('int32'))
                
                if c == 0: # mask
                    color = (0,250,154)
                else: # no mask
                    color = (255, 0, 0)
                    
                draw = ImageDraw.Draw(frame)
                
                for i in range(2):
                    draw.rectangle((left+i,top+i,right-i,bottom-i),outline=color,width=5)
                    
                del draw
                
        return frame                





if __name__ == '__main__':
    
    x = torch.rand(1,3,416,416)
    model = Yolo(3,3,2)
    model.detect(x)
    #odel = Yolo(3,20,5)
    #out1,out2 = model(x)

    #print('out1 :',out1.shape,'out2:',out2.shape)


        

