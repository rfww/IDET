
import cv2
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.read_image import read_image
EXTENSIONS = ['.jpg', '.png','.JPG','.PNG']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, '{}{}'.format(basename,extension))

def image_path_city(root, name):
    return os.path.join(root, '{}'.format(name))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class TrainData(Dataset):
    def __init__(self, imagepath=None, imagepath2=None, labelpath=None, transform=None):
        #  make sure label match with image 
        self.transform = transform 
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(imagepath2), "{} not exists !".format(imagepath2)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)                                  
        
        image  = read_image(imagepath)
        image2 = read_image(imagepath2)
        label  = read_image(labelpath)
        self.train_set = (
            image,
            image2,
            label
        )
       

    def __getitem__(self, index):
        filename   = self.train_set[0][index]
        filename2  = self.train_set[1][index]
        filenameGt = self.train_set[2][index]
        # roiname = os.path.join(os.path.split(filename)[0].replace("input", ""), "ROI.bmp")
        # print(filename)
        with open(filename, 'rb') as f: 
            image = load_image(f).convert('RGB')
        with open(filename2, 'rb') as f:
            image2 = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
        # with open(roiname, 'rb') as f:
        #     roi = load_image(f).convert('1')
        # roi = Image.fromarray(cv2.imread(roiname))

        if self.transform is not None:#########################
            image, image2, label = self.transform(image, image2, label)

        return image, image2, label

    def __len__(self):
        return len(self.train_set[0])
    
class TestData(Dataset):
    def __init__(self, imagepath=None, imagepath2=None, labelpath=None, transform=None):
        self.transform = transform 
        
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(imagepath2), "{} not exists !".format(imagepath2)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        
        image  = read_image(imagepath)
        image2 = read_image(imagepath2)
        label  = read_image(labelpath)
        self.test_set = (
            image,
            image2,
            label
        )
        print("Length of test data is {}".format(len(self.test_set[0])))
    def __getitem__(self, index):
        filename   = self.test_set[0][index]
        filename2  = self.test_set[1][index]
        filenameGt = self.test_set[2][index]

        
        with open(filename, 'rb') as f: # advance
            image = load_image(f).convert('RGB')
        with open(filename2,'rb') as f:
            image2 = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
        # with open(roiname, 'rb') as f:  # roi
        #     roi = load_image(f).convert('1')
        # roi = Image.fromarray(cv2.imread(roiname))

        if self.transform is not None:
            image_tensor, image_tensor2, label_tensor, img, img2 = self.transform(image, image2, label)
            return (image_tensor, image_tensor2, label_tensor,filenameGt, np.array(img),np.array(img2))
        return np.array(image), np.array(image2),filenameGt

    def __len__(self):
        return len(self.test_set[0])

