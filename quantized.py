from utils2 import cells_to_bboxes, non_max_suppression, use_gpu_if_possible
from backbone_quant import backbone
from CSP_quant import ConvBlock
import torch.nn as nn
import torch
from train2 import test_model 
import os
from loss import Loss
from utils2 import  AverageMeter, class_accuracy
from dataset import get_data
import torch.optim as optim
import numpy as np

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v




# define a floating point model where some layers could benefit from QAT
class Yolo_Q(nn.Module):
    
    def __init__(self,in_channels,B,num_classes):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.back = backbone(in_channels)
        self.conv1 = ConvBlock(512,512,3,1)
        self.conv2 = ConvBlock(512,256,3,1)
        self.conv3 = nn.Conv2d(512,128,1,1)
        self.upsample = nn.ConvTranspose2d(128,256,2,2)
        self.conv4 = nn.Conv2d(256,255,1,1)
        self.conv5 = nn.Conv2d(512,255,1,1)
        self.head = nn.Conv2d(255,B*(5+num_classes),1,1)
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

        self.B = B
        
    def forward(self,x):
        x = self.quant(x)
        out1 , out2 = self.back(x)
        out2 = self.conv1(out2)
        feat2 = out2
        out2 = self.conv3(out2)
        feat1 = torch.cat([out1,self.upsample(out2)],dim=1)
        feat2 = self.conv1(feat2)
        feat1 = self.conv2(feat1)
        feat1 = self.conv4(feat1)
        feat2 = self.conv5(feat2)

        f2 = self.dequant(self.head(feat2))
        f1 = self.dequant(self.head(feat1))

        out = f2.reshape(feat2.shape[0], self.B, 2 + 5, feat2.shape[2], feat2.shape[3]).permute(0, 1, 3, 4, 2),f1.reshape(feat1.shape[0], self.B, 2 + 5, feat1.shape[2], feat1.shape[3]).permute(0, 1, 3, 4, 2)
        return out
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBlock:
                torch.quantization.fuse_modules(m, ['conv', 'bn'], inplace=True)


def load_model(model_file):
    model = Yolo_Q(3,3,2)
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def evaluate(model, data_loader, neval_batches):
    model.eval()
    
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            
            if cnt >= neval_batches:
                 return top1, top5


            
            



if __name__ == '__main__':
    
    num_calibration_batches = 32
    train_loader, test_loader = get_data('train.csv','test.csv')
    print("loaded data")
    model = load_model("model_newdata.pt").to('cpu')
    print_size_of_model(model)
    model.eval()
    
    model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    #randinput = torch.randn(1, 3, 416, 416)
    #model(randinput)
    evaluate(model, train_loader, neval_batches=num_calibration_batches)

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')

    print("Size of model after quantization")
    print_size_of_model(model)

    
    S=[13, 26]
    ANCHORS =  [[(0.275 ,   0.320312), (0.068   , 0.113281), (0.017  ,  0.03   )], 
               [(0.03  ,   0.056   ), (0.01  ,   0.018   ), (0.006 ,   0.01    )]]

    scaled_anchors = (
            torch.tensor(ANCHORS)
            * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to("cpu")

    optimizer = optim.SGD(
            model.parameters(), lr=0.001, weight_decay=0.0005
        )
    loss_fn = Loss()


    test_model(model, test_loader, scaled_anchors, performance=class_accuracy, loss_fn= Loss(), device='cpu')

    torch.jit.save(torch.jit.script(model), "models/quantized.pt" )