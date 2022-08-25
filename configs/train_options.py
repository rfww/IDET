# -*- coding:utf-8 -*-
import argparse
import os


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--model', default="idet", help='model to train,options:fcn8,segnet...')
        self.parser.add_argument('--state')
        self.parser.add_argument('--num-classes', type=int, default=2)
        self.parser.add_argument('--datadir', default="./data_CDnet/", help='path for training data')
        self.parser.add_argument('--savedir', type=str, default='./save_models2022/IDET_CDnet/', help='savedir for models')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--num-epochs', type=int, default=20)
        self.parser.add_argument('--num-workers', type=int, default=2)
        self.parser.add_argument('--batch-size', type=int, default=2)
        self.parser.add_argument('--epoch-save', type=int,
                                 default=10)  # You can use this value to save model every X epochs
        self.parser.add_argument('--iouTrain', action='store_true',
                                 default=False)  # recommended: False (takes a lot to train otherwise)
        self.parser.add_argument('--steps-loss', type=int, default=100)
        self.parser.add_argument('--pretrained', type=str, default='')
        self.parser.add_argument('--local_rank', default=-1, type=int, help='node rank of distributed training')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
