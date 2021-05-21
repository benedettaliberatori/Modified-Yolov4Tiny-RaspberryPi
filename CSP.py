import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activat = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activat(x)
        return x

    
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels,out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels // 2, 3)
        self.conv3 = ConvBlock(out_channels // 2, out_channels // 2, 3)
        self.conv4 = ConvBlock(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        feat = x
        c = self.out_channels
        #x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        feat1 = x
        x = self.conv3(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.conv4(x)
        feat1 = x
        x = torch.cat([feat, x], dim=1)
        #feat2 = self.maxpool(x)

        return x



class ResBlockD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,stride = 1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride =2,padding=1)
        self.conv3 = nn.Conv2d(out_channels,in_channels,1,stride =1)
        self.conv4 = nn.Conv2d(in_channels,in_channels,1,stride =1)
        self.avgpool = nn.AvgPool2d(2,2)

    def forward(self,x):
        x_a = self.conv1(x)
        x_b = self.avgpool(x)
        x_a = self.conv2(x_a)
        x_a = self.conv3(x_a)
        x_b = self.conv4(x_b)

        out = torch.cat([x_a,x_b],dim=1)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // ratio, bias=False),
                               nn.ReLU(),
                               nn.Linear(in_channels // ratio, in_channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        avg = avg.view(-1,1*1*32)

        avg_out = self.fc(avg)
        max = self.max_pool(x)
        max = avg.view(-1,1*1*32)
        
        max_out = self.fc(max)
        avg_out = avg_out.view(1,32,1,1)
        max_out = avg_out.view(1,32,1,1)
        out = avg_out + max_out
        return self.sigmoid(out)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AuxiliaryResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self,x):

        x = self.conv1(x)
        feat = x
        x = self.conv2(x)
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        out = torch.cat([x,feat],dim=1)

        return out


    


if __name__ == '__main__':
    x = torch.rand(1,64,104,104)
    #model = ResBlockD(64,32)
    #print(model(x).shape)
    
    model2 = AuxiliaryResBlock(64,32)
    print(model2(x).shape)
    
    #model3 = AuxiliaryResBlock(64, 32)
    #print(model3(x).shape)



