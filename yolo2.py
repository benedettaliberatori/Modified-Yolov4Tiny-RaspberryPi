from utils2 import cells_to_bboxes, non_max_suppression, use_gpu_if_possible
from backbone import backbone
from CSP import ConvBlock
import torch.nn as nn
import torch
from torchvision import transforms
import cv2

#class DecodeBox(nn.Module):
#
#    def __init__(self, scaled_anchors, num_classes, img_size):
#        super(DecodeBox, self).__init__()
#        self.scaled_anchors = scaled_anchors
#        self.num_anchors = len(scaled_anchors)
#        self.num_classes = num_classes
#        self.bbox_attrs = 5 + num_classes
#        self.img_size = img_size
#
#    def forward(self,input):
#
#        batch_size = input.size(0)
#        input_height = input.size(2)
#        input_width = input.size(3)
#        stride_h = self.img_size[1] / input_height
#        stride_w = self.img_size[0] / input_width
#
#        prediction = input
#
#        x = torch.sigmoid(prediction[..., 0])
#        y = torch.sigmoid(prediction[..., 1])
#        w = prediction[..., 2]
#        h = prediction[..., 3]
#        conf = torch.sigmoid(prediction[..., 4])
#        pred_cls = torch.sigmoid(prediction[..., 5:])
#        FloatTensor = torch.FloatTensor
#        LongTensor = torch.LongTensor
#
#        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
#            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
#        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
#            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
#        anchor_w = FloatTensor(self.scaled_anchors).index_select(1, LongTensor([0]))
#        anchor_h = FloatTensor(self.scaled_anchors).index_select(1, LongTensor([1]))
#        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
#        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
#        pred_boxes = FloatTensor(prediction[..., :4].shape)
#
#        pred_boxes[..., 0] = x.data + grid_x
#        pred_boxes[..., 1] = y.data + grid_y
#        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
#        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
#        scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
#        output = torch.cat((pred_boxes.reshape(batch_size, -1, 4)*scale,
#                            conf.reshape(batch_size, -1, 1), pred_cls.reshape(batch_size, -1, self.num_classes)), -1)
#        return output.data
#




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
        self.net=Yolo_Block(3,3,2).eval()
        
        model_dict=torch.load("pruned_RAdam_5.pt", map_location = use_gpu_if_possible())
        self.net.load_state_dict(model_dict)
        

    def detect_Persson(self, CV2_frame,Tensor_frame, scaled_anchors, iou_thresh = .3, tresh = .65 ):
                       
        with torch.no_grad():

            out = self.net(Tensor_frame)
            boxes = []
            
            for i in range(2):
                anchor = scaled_anchors[i]
                #print(anchor.shape)
                #print(out[i].shape)
                boxes += cells_to_bboxes(out[i], S=out[i].shape[2], anchors = anchor)[0]
                
            boxes = non_max_suppression(boxes, iou_threshold= iou_thresh, threshold=tresh, box_format = "midpoint")
            

            for box in boxes:
                if box[0] == 0: # mask
                        color = (0,250,154)
                        label = 'mask'
                else: # no mask
                        color = (255, 0, 0)
                        label = 'no mask'
                height, width = 416, 416

                box = box[2:]
                p0 = (int((box[0] - box[2]/2)*height) ,int((box[1] - box[3]/2)*width))
                p1 = (int((box[0] + box[2]/2)*height) ,int((box[1] + box[3]/2)*width))
                
                #print(p0)
                #print(p1)
                
                CV2_frame = cv2.rectangle(CV2_frame, p0, p1, color, thickness=2)
                cv2.putText(CV2_frame, label + "{:.2f}".format(box[1]*100) + '%', (int((box[0] - box[2]/2)*height), int((box[1] - box[3]/2)*width)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            return CV2_frame           


if __name__ == '__main__':
     
    ANCHORS =  [[(0.275 ,   0.320312), (0.068   , 0.113281), (0.017  ,  0.03   )], 
           [(0.03  ,   0.056   ), (0.01  ,   0.018   ), (0.006 ,   0.01    )]]  
    S = [13,26]   
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        
    model = Yolo()
    
    image = cv2.imread("arianna.jpg", cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.ToTensor()(image).unsqueeze_(0)
    
    image = model.detect_Persson(image,image_tensor,scaled_anchors)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("new_arianna.jpg", image)
    