from CSP_quant import AuxiliaryResBlock,ResBlockD,CSPBlock,ConvBlock
import torch.nn as nn
import torch

class backbone(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels,32,3,2)
        self.conv2 = ConvBlock(32,64,3,2)
        self.resblock1 = ResBlockD(64,128)
        self.auxiliary1 = AuxiliaryResBlock(128)
        self.auxiliary2 = AuxiliaryResBlock(256)
        self.resblock2 = ResBlockD(128,256)
        self.csp = CSPBlock(256,256)
        self.conv3 = ConvBlock(512,512,3,1)
        self.convaux = ConvBlock(128,256,3,2)
        self.ff = torch.nn.quantized.FloatFunctional()
    def forward(self,x):
        x = self.resblock1(self.conv2(self.conv1(x)))
        out = self.auxiliary1(x)
        x = self.resblock2(self.ff.add(x,out))
        feat = x
        out = self.convaux(out)
        out = self.auxiliary2(out)
        x = self.csp(self.ff.add(x,out))
        x = self.conv3(x)

        return feat , x 




if __name__ == '__main__':

    x = torch.rand(1,3,416,416)
    model = backbone(3)
    out1,out2 = model(x)

    print('out1 :',out1.shape,'out2:',out2.shape)
