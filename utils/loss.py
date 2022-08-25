import torch.nn as nn
import torch


def cd_loss(input,target):
    # input = torch.tensor(input.float)
    # target = torch.tensor(target.float)
    # input = torch.tensor(input.float)
    # target = torch.tensor(target.float)
    # print(input.shape)
    # print(target.shape)
   
    bce_loss = nn.BCELoss()
    bce_loss = bce_loss(torch.sigmoid(input.float()),torch.sigmoid(target.float()))
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dic_loss = 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

    return  dic_loss + bce_loss
