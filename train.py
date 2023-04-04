import os
import time
import math

import torch
import numpy as np

from torchvision import models
from eval import *
import torch.nn as nn
from utils import evalIoU
from models import get_model
from torch.autograd import Variable
from dataloader.dataset import TrainData
from torch.utils.data import DataLoader
from dataloader.transform import MyTransform
from torchvision.transforms import ToPILImage
from configs.train_options import TrainOptions
from torch.optim import SGD, Adam, lr_scheduler
from criterion.criterion import CrossEntropyLoss2d
import argparse
from tqdm import tqdm


NUM_CHANNELS = 3

def get_loader(args):

    imagepath_train = os.path.join(args.datadir, 'train/image.txt')
    imagepath_train2 = os.path.join(args.datadir, 'train/image2.txt')
    labelpath_train = os.path.join(args.datadir, 'train/label.txt')

    #train_transform = MyTransform(reshape_size=(256, 256), crop_size=(256, 256), # remote sensing scencs
    train_transform = MyTransform(reshape_size=(320, 320), crop_size=(320, 320),  # street views
                                  augment=True)  # data transform for training set with data augmentation, including resize, crop, flip and so on
    dataset_train = TrainData(imagepath_train, imagepath_train2, labelpath_train, train_transform)  # DataSet
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return loader

def print_model(model):  # compute the number of parameters
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("Total number of parameters: {}M".format(num_params/1048576))
    return

def train(args, model):
    NUM_CLASSES = args.num_classes  # pascal=21, cityscapes=20

    
    savedir = args.savedir
    weight = torch.ones(NUM_CLASSES)
    loader = get_loader(args)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight).cuda()
    else:
        criterion = CrossEntropyLoss2d(weight)

    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTrain-IoU\t\tlearningRate")
    paras = dict(model.named_parameters())
    paras_new = []

    for k, v in paras.items():

        if 'bias' in k:
            if 'dec' in k:
                paras_new += [{'params': [v], 'lr': 0.02 * args.lr, 'weight_decay': 0}]
            else:
                paras_new += [{'params': [v], 'lr': 0.2 * args.lr, 'weight_decay': 0}]
        else:
            if 'dec' in k:
                paras_new += [{'params': [v], 'lr': 0.01 * args.lr, 'weight_decay': 0.00004}]
            else:
                paras_new += [{'params': [v], 'lr': 0.1 * args.lr, 'weight_decay': 0.00004}]
    optimizer = Adam(paras_new, args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # learning rate changed every epoch
    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs + 1):
        tbar = tqdm(loader, desc='\r')
        scheduler.step(epoch)
        epoch_loss = []
        time_train = []
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)

        usedLr = 0
        # for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            usedLr = float(param_group['lr'])

        model.cuda().train()
        for step, (images, images2, labels) in enumerate(tbar):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()
            inputs = Variable(images)
            inputs2 = Variable(images2)
            targets = Variable(labels)
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = model(inputs, inputs2)
            loss = criterion(p1, targets[:, 0])
            loss1 = criterion(p2, targets[:, 0])
            loss2 = criterion(p3, targets[:, 0])
            loss3 = criterion(p4, targets[:, 0])
            loss4 = criterion(p5, targets[:, 0])
            loss5 = criterion(p6, targets[:, 0])
            loss6 = criterion(p7, targets[:, 0])
            loss7 = criterion(p8, targets[:, 0])
            loss8 = criterion(p9, targets[:, 0])
            loss9 = criterion(p10, targets[:, 0])
            loss10= criterion(p11, targets[:, 0])
            loss11= criterion(p12, targets[:, 0])


            loss += loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                localtime = time.asctime(time.localtime(time.time()))
                tbar.set_description('loss: %.8f | epoch: %d | step: %d | Time: %s' % (average, epoch, step, str(localtime)))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouAvgStr, iouTrain, classScoreList = cal_iou(evalIoU, confMatrix)


        if epoch % args.epoch_save == 0:
           torch.save(model.state_dict(), '{}_{}.pth'.format(os.path.join(args.savedir, args.model), str(epoch)))

        # save log
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, iouTrain, usedLr))

    return ''


def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = '{}'.format(args.savedir)
    modeltxtpath = os.path.join(savedir, 'model.txt')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    #torch.backends.cudnn.enabled = False
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile:  # record options
        myfile.write(str(args))

    # initialize the network
    model = get_model(args)  # load model
    decoders = list(models.vgg16_bn(pretrained=True).features.children())
    model.dec1 = nn.Sequential(*decoders[:7])
    model.dec2 = nn.Sequential(*decoders[7:14])
    model.dec3 = nn.Sequential(*decoders[14:24])
    model.dec4 = nn.Sequential(*decoders[24:34])
    model.dec5 = nn.Sequential(*decoders[34:44])
    
    print_model(model)  # print the number of parameters

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True

    with open(modeltxtpath, "w") as myfile:  # record model
        myfile.write(str(model))

    model = model.to(device)
    train(args, model)
    
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = TrainOptions().parse()
    main(parser)
