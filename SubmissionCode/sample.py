import os
import random
import argparse
import pandas as pd
import numpy as np

import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import shutil


# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    print(bb)
    bb = np.array(bb, dtype=np.float32)
    print(bb)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

"""Resizing image and bounding boxes"""

#Reading an image
def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 21))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]

class RoadDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = int(self.y[idx])
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms)
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb

def create_corner_rect(bb,img,label,init_folder, color='red'):
    print(bb)
    bb = np.array(bb, dtype=np.float32)
    print(bb)
    print(label)
    _ = cv2.rectangle(img, (bb[1], bb[0]), (bb[3], bb[2]), (255,0,0), 2)
    cv2.putText(img, str(label), (int(bb[1]), int(bb[0]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imwrite('./'+init_folder+'/predicted.png',img)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im,label, bb, init_folder):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb,im, label, init_folder))

def main(args):
    
    print("initialize parameter")
    image_path = args.image
    class_dict = {'fries': 0, 'omelette': 1, 'rice': 2, 'icecream': 3, 'apple_pie': 4, 'burger': 5, 'hotdog': 6, 'beer': 7, 'bolonese_pasta': 8, 'bread': 9, 'carbonara': '10', 'carbonara_pasta': 11, 'chantilly': 12, 'coffee': 13
    ,'cream': 14, 'fires': 15, 'ketchup': 16, 'pie': 17, 'potatoes': 18, 'salad': 19, 'sandwich': 20}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_folder = './results'

    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    else:
        shutil.rmtree(init_folder)
        os.makedirs(init_folder)


    print("setting up model")
    model = BB_model().to(device)
    para = torch.load("model_parameter.pt")
    model.load_state_dict(para)

    im = read_image(image_path)
    im = cv2.resize(im, (int(1.49*300), 300))
    cv2.imwrite(image_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    
    """Make Prediction"""
    print("prepping")
    test_ds = RoadDataset(pd.DataFrame([{'path':image_path}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':0}])['y'])
    x, y_class, y_bb = test_ds[0]

    xx = torch.FloatTensor(x[None,])

    # prediction
    model.eval( )
    out_class, out_bb = model(xx.cuda())

    # predicted class
    torch.max(out_class, 1)
    print(list(class_dict.keys())[list(class_dict.values()).index(torch.argmax(out_class))])

    # predicted bounding box
    label = list(class_dict.keys())[list(class_dict.values()).index(torch.argmax(out_class))]
    bb_hat = out_bb.detach().cpu().numpy()
    bb_hat = bb_hat.astype(int)
    show_corner_bb(im, label,bb_hat[0], init_folder)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # image path
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    args = parser.parse_args()
    main(args)
