import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models

class FCN8_new(nn.Module):

    def __init__(self, num_classes):

        super().__init__()
        my_model = models.vgg16(pretrained=True)
        input_1_new = nn.Conv2d(6, 64, (3, 3), 1, 1)
        my_model.features[0] = input_1_new
        feats = list(my_model.features.children())
        self.feats = nn.Sequential(*feats[0:10])
        self.feat3 = nn.Sequential(*feats[10:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:31])

        #for m in self.modules():
         #   if isinstance(m, nn.Conv2d):
          #      m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x,y):
            concat_xy = torch.cat([x,y],1)
            feats = self.feats(concat_xy)
            feat3 = self.feat3(feats)
            feat4 = self.feat4(feat3)
            feat5 = self.feat5(feat4)
            fconn = self.fconn(feat5)

            score_feat3 = self.score_feat3(feat3)
            score_feat4 = self.score_feat4(feat4)
            score_fconn = self.score_fconn(fconn)

            score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
            score += score_feat4
            score = F.upsample_bilinear(score, score_feat3.size()[2:])
            score += score_feat3

            return F.upsample_bilinear(score, x.size()[2:])

#model = FCN8()
#print(model)
#print(model.score_feat4)
#print(model.feats[0])
#print(model.input_1)

#model_2 = models.vgg16(pretrained=False)
#print(model_2)
#input_1 = model_2.features[0]
#input_1_new = nn.Conv2d(6,64,(3,3),1,1)
#model_2.features[0] = input_1_new
#print(input_1_new)
#print(model_2)