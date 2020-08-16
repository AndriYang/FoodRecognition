# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:23:57 2020
@author: Asus
"""

#library imports
import os
import random
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import shutil

#Provide list of filenames under root directory
def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def generate_train_df (anno_path, images_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        anno['class'] = root.find("./object/name").text
        anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
        anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
        anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
        anno['ymax'] = int(root.find("./object/bndbox/ymax").text)
        anno_list.append(anno)
    return pd.DataFrame(anno_list)

def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
#Make a mask for the bounding box of image
def create_mask(bb, x):
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    return np.array([x[5],x[4],x[7],x[6]])

def resize_image_bb(read_path,write_path,bb,sz):
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

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
    x = cv2.imread(str(path))
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

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))

def normalise(image):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (image - imagenet_stats[0]) / imagenet_stats[1]
        
def plot_loss_graph(epoch, training_losses, validation_losses, init_folder):
    epoch_list = np.arange(1, epoch + 1)
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, training_losses, label = "Training loss")
    plt.plot(epoch_list, validation_losses, label = "Validation loss")
    plt.legend(loc = "upper right")
    directory = str(init_folder) + "/loss.png"
    plt.savefig(directory)
    plt.show()
    
class FoodDataset(Dataset):
    def __init__(self, dirs, bounding_box, y, transforms=False):
        self.transforms = transforms
        self.dirs = dirs.values
        self.bounding_box = bounding_box.values
        self.y = y.values
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, idx):
        directory = self.dirs[idx]
        y_class = int(self.y[idx])
        x, y_bb = transformsXY(directory, self.bounding_box[idx], self.transforms)
        x = normalise(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb

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
    

def train_epocs(model, optimizer, train_dl, val_dl, epochs=10,C=1000):
    idx = 0
    val_losses=[]
    train_losses=[]

    for i in range(epochs):
        print('Epoch {}/{}'.format(i + 1, epochs))
        print('-' * 10)

        model.train()
        total = 0
        sum_loss = 0
        for idx, batch in enumerate(train_dl):
            x = batch[0]
            y_class = batch[1]
            y_bounding_box = batch[2]
            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_bounding_box = y_bounding_box.to(device).float()
            out_class, out_bounding_box = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bounding_box = F.l1_loss(out_bounding_box, y_bounding_box, reduction="none").sum(1)
            loss_bounding_box = loss_bounding_box.sum()
            loss = loss_class + loss_bounding_box/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        train_losses.append(train_loss)
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))

        if i == 0:
            torch.save(model.state_dict(), "model_parameter.pt")
            print("At epoch {}, save model with lowest validation loss: {}".format(i, val_loss))

        if (len(val_losses) > 0) and (val_loss < min(val_losses)):
            torch.save(model.state_dict(), "model_parameter.pt")
            print("At epoch {}, save model with lowest validation loss: {}".format(i, val_loss))
        val_losses.append(val_loss)
    return train_losses, val_losses
    
def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for idx, batch in enumerate(valid_dl):
        x = batch[0]
        y_class = batch[1]
        y_bounding_box = batch[2]
        batch = y_class.shape[0]
        x = x.to(device).float()
        y_class = y_class.to(device)
        y_bounding_box = y_bounding_box.to(device).float()
        out_class, out_bounding_box = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bounding_box = F.l1_loss(out_bounding_box, y_bounding_box, reduction="none").sum(1)
        loss_bounding_box = loss_bounding_box.sum()
        loss = loss_class + loss_bounding_box/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    val_loss = sum_loss/total
    val_acc = correct/total
    return val_loss, val_acc

if __name__ == '__main__':
    images_dir = Path('./food/images')
    annotation_dir = Path('./food/annotations')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 2
    batch_size = 4
    test_size = 0.2
    random_state = 42
    lr = 0.001
    init_folder = './results'
    
    resize_folder =  'food/images_resized'
    signs_test_folder = 'food/food_test'
    
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    else:
        shutil.rmtree(init_folder)
        os.makedirs(init_folder) 

    if not os.path.exists(resize_folder):
        os.makedirs(resize_folder)
    else:
        shutil.rmtree(resize_folder)
        os.makedirs(resize_folder)
        
    if not os.path.exists(signs_test_folder):
        os.makedirs(signs_test_folder)
    else:
        shutil.rmtree(signs_test_folder)
        os.makedirs(signs_test_folder)
        
    df_train = generate_train_df(annotation_dir, images_dir)

    #label encode target
    class_dict = {'fries': 0, 'omelette': 1, 'rice': 2, 'icecream': 3, 'apple_pie': 4, 'burger': 5, 'hotdog': 6, 'beer': 7, 'bolonese_pasta': 8, 'bread': 9, 'carbonara': '10', 'carbonara_pasta': 11, 'chantilly': 12, 'coffee': 13,'cream': 14, 'fires': 15, 'ketchup': 16, 'pie': 17, 'potatoes': 18, 'salad': 19, 'sandwich': 20}
    df_train['class'] = df_train['class'].apply(lambda x:  class_dict[x])
    
    #Populating Training DF with new paths and bounding boxes
    new_paths = []
    new_bbs = []
    train_path_resized = Path('./food/images_resized')
    for index, row in df_train.iterrows():
        new_path,new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values),300)
        new_paths.append(new_path)
        new_bbs.append(new_bb)
    df_train['new_dir'] = new_paths
    df_train['new_bounding_box'] = new_bbs
    
    X = df_train[['new_dir', 'new_bounding_box']]
    Y = df_train['class']
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    
    train_ds = FoodDataset(X_train['new_dir'],X_train['new_bounding_box'] ,y_train, transforms=True)
    valid_ds = FoodDataset(X_val['new_dir'],X_val['new_bounding_box'],y_val)
    
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    
    model = BB_model().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    train_losses, v_losses = train_epocs(model, optimizer, train_dl, valid_dl, epochs)  
    
    plot_loss_graph(epochs, train_losses, v_losses, init_folder)
