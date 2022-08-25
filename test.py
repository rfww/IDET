import os
import time
import torch
from configs.test_options import TestOptions
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataloader.transform import Transform_test
from dataloader.dataset import TestData
from models import get_model
import copy


def main(args):
    despath = args.savedir
    if not os.path.exists(despath):
        os.makedirs(despath)
        # os.mkdir(despath)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    imagedir = os.path.join(args.datadir, 'image.txt')
    image2dir = os.path.join(args.datadir, 'image2.txt')
    labeldir = os.path.join(args.datadir, 'label.txt')
                                         
    transform = Transform_test(args.size)
    dataset_test = TestData(imagedir, image2dir, labeldir, transform)
    loader = DataLoader(dataset_test, num_workers=4, batch_size=1, shuffle=False)  # test data loader
    model = get_model(args)

    if args.cuda:
        model = model.cuda()

    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)
    model.eval()
    count = 0

    for step, colign in enumerate(loader):

        images = colign[0]
        images2 = colign[1]
        # label = colign[2]
        file_name = colign[3]
        # ----------street view datasets:-------
        image_name = file_name[0].split("/")[-1]
        folder_name = file_name[0].split("/")[-3]
        #---------------------------------------

        #---------remote sensing dataset--------
        basename = os.path.basename(file_name)
        #---------------------------------------
        if args.cuda:
            images = images.cuda()
            images2 = images2.cuda()


        inputs = Variable(images, volatile=True)
        inputs2 = Variable(images2, volatile=True)

        _,_,_,_,_,_,pf,_,_,_,_,_ = model(inputs, inputs2)

        out_p = pf[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
        if "CDnet" in args.datadir:
            # image_name = file_name[0].split("/")[-1]
            pfolder_name = file_name[0].split("/")[-4]
            # folder_name = file_name[0].split("/")[-3]
            if not os.path.exists(despath + pfolder_name + '/'):
                os.makedirs(despath + pfolder_name + '/')
            if not os.path.exists(despath + pfolder_name + '/' + folder_name):
                os.makedirs(despath + pfolder_name + '/' + folder_name)
            Image.fromarray(np.uint8(out_p * 255)).save(
                despath + pfolder_name + '/' + folder_name + '/' + image_name.split(".")[0] + '.png')
        elif "CMU" in args.datadir or "PCD" in args.datadir:
            Image.fromarray(np.uint8(out_p * 255)).save(despath + folder_name + '_' + image_name.split(".")[0] + '.png')
        else:
            Image.fromarray(np.uint8(out_p * 255)).save(despath + basename)  # remote sensing datasets


        print("This is the {}th of image!".format(count))



if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)


