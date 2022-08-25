import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class Geneator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1, padding=0),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),  
            
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 4, 1, padding=0),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),  
            
            nn.Conv2d(in_channels * 4, in_channels * 4, 3, padding=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 8, 1, padding=0),
            nn.BatchNorm2d(in_channels * 8),
            nn.ReLU(inplace=True),  
            
            nn.Conv2d(in_channels * 8, in_channels * 8, 3, padding=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(in_channels * 8),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out1, out2, out3

class CCP_Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.sgp= SegNetEnc(in_dim*(2+4+8), out_dim,1,1)
        self.ccp_G = Geneator(in_dim)

        self.pr1 = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim*2,3,1,1),
            nn.BatchNorm2d(in_dim*2),
            nn.ReLU(inplace=True)
        )
        self.pr2 = nn.Sequential(
            nn.Conv2d(in_dim*4,in_dim*4,3,1,1),
            nn.BatchNorm2d(in_dim*4),
            nn.ReLU(inplace=True)
        )
        self.pr3 = nn.Sequential(
            nn.Conv2d(in_dim*8,in_dim*8,3,1,1),
            nn.BatchNorm2d(in_dim*8),
            nn.ReLU(inplace=True)
        )
      
    def forward(self, x, y):
    
        x1, x2, x3 = self.ccp_G(x)
        y1, y2, y3 = self.ccp_G(y)

        pr1 = self.pr1(abs(x1-y1))
        pr2 = self.pr2(abs(x2-y2))
        pr3 = self.pr3(abs(x3-y3))
        ccp = self.sgp(torch.cat([
            F.upsample_bilinear(pr1, scale_factor=0.5),
            pr2,
            F.upsample_bilinear(pr3, scale_factor=2)
        ],1))
       
        return ccp
