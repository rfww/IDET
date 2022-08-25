import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
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

class up_layer(nn.Module):
    def __init__(self,scale):
        super().__init__()

        self.up_layer = nn.Sequential(nn.Upsample(scale_factor=scale, mode='bilinear'))
    def forward(self,x):
        return self.up_layer(x)

class predictor(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.enc1 = SegNetEnc(128, 64, 1)
        self.up = up_layer(2)
        self.enc2 = SegNetEnc(128, 64, 1)
        self.predictor = nn.Conv2d(64, num_classes, 3, padding=1)
    def forward(self, feat_low, feat_high):
        enc_feat_high = self.enc1(feat_high)
        enc_feat_high_up = self.up(enc_feat_high)
        enc_feat_hl = self.enc2(torch.cat([enc_feat_high_up, feat_low], 1))
        prediction = self.predictor(enc_feat_hl)
        return prediction



class eNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        #decoders = list(models.vgg16(pretrained=True).features.children())
        my_model = models.vgg16(pretrained=True)
        input_1_new = nn.Conv2d(6, 64, (3, 3), 1, 1)
        my_model.features[0] = input_1_new
        feats = list(my_model.features.children())
        self.feat1 = nn.Sequential(*feats[0:5])
        self.feat2 = nn.Sequential(*feats[5:10])
        self.feat3 = nn.Sequential(*feats[10:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:31])

        # for m in self.modules():
        #   if isinstance(m, nn.Conv2d):
        #      m.requires_grad = False

        self.enc5 = SegNetEnc(512, 64, 1)
        self.enc4D = nn.Conv2d(512, 64, 3, padding=1)
        self.enc4 = SegNetEnc(128, 64, 1)
        self.enc3D = nn.Conv2d(256, 64, 3, padding=1)
        self.enc3 = SegNetEnc(128, 64, 1)
        self.enc2D = nn.Conv2d(128, 64, 3, padding=1)
        #self.enc2 = SegNetEnc(256, 64, 1)
        #self.enc1 = SegNetEnc(128, 64, 1)

        self.enc5_up = up_layer(2)
        self.enc4_up = up_layer(2)
        self.enc3_up = up_layer(2)
        #self.enc2_up = up_layer(2)

        # predictors
        self.p1 = predictor(num_classes)
        self.p2 = predictor(num_classes)
        self.p3 = predictor(num_classes)
        self.p4 = predictor(num_classes)
        self.p5 = predictor(num_classes)


    def forward(self, x, y):
        '''
            Attention, input size should be the 32x.
        '''
        dec1 = self.feat1(torch.cat([x, y], 1))
        dec2 = self.feat2(dec1)
        dec3 = self.feat3(dec2)
        dec4 = self.feat4(dec3)
        dec5 = self.feat5(dec4)
        enc5 = self.enc5(dec5)
        dec4d = self.enc4D(dec4)
        dec3d = self.enc3D(dec3)
        dec2d = self.enc2D(dec2)

        enc5_up = self.enc5_up(enc5)
        enc4 = self.enc4(torch.cat([enc5_up, dec4d], 1))
        enc4_up = self.enc4_up(enc4)
        enc3 = self.enc3(torch.cat([enc4_up, dec3d], 1))
        enc3_up = self.enc3_up(enc3)
        enc2 = torch.cat([enc3_up, dec2d], 1)

        p1 = self.p1(dec1, enc2)
        p1f = F.upsample_bilinear(p1, x.size()[2:])
        p2 = self.p2(dec1, enc2)
        p2f = F.upsample_bilinear(p2, x.size()[2:])
        p3 = self.p3(dec1, enc2)
        p3f = F.upsample_bilinear(p3, x.size()[2:])
        p4 = self.p4(dec1, enc2)
        p4f = F.upsample_bilinear(p4, x.size()[2:])
        p5 = self.p5(dec1, enc2)
        p5f = F.upsample_bilinear(p5, x.size()[2:])

        return p1f, p2f, p3f, p4f, p5f
