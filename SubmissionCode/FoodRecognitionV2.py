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
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def produce_train_df (annotation_dir):
    annotations = filelist(annotation_dir, '.xml')
    annotation_list = []
    for annotation_dir in annotations:
        root = ET.parse(annotation_dir).getroot()
        annotation = {}
        annotation['filename'] = Path(str(images_dir) + '/'+ root.find("./filename").text)
        annotation['width'] = root.find("./size/width").text
        annotation['height'] = root.find("./size/height").text
        annotation['class'] = root.find("./object/name").text
        annotation['xmin'] = int(root.find("./object/bndbox/xmin").text)
        annotation['ymin'] = int(root.find("./object/bndbox/ymin").text)
        annotation['xmax'] = int(root.find("./object/bndbox/xmax").text)
        annotation['ymax'] = int(root.find("./object/bndbox/ymax").text)
        annotation_list.append(annotation)
    return pd.DataFrame(annotation_list)

def read_image(directory):
    return cv2.cvtColor(cv2.imread(str(directory)), cv2.COLOR_BGR2RGB)

#Make a mask for the bounding box of image
def make_mask(bounding_box, x):
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bounding_box = bounding_box.astype(np.int)
    Y[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]] = 1.
    return Y

#Make mask Y to a bounding box
def mask_to_bounding_box(Y):
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    row_top = np.min(rows)
    col_left = np.min(cols)
    row_bottom = np.max(rows)
    col_right = np.max(cols)
    return np.array([col_left, row_top, col_right, row_bottom], dtype=np.float32)

def generate_bounding_box_array(input):
    return np.array([input[5],input[4],input[7],input[6]])

def resize_image_bounding_box(read_dir, write_dir, bounding_box, sz):
    im = read_image(read_dir)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(make_mask(bounding_box, im), (int(1.49*sz), sz))
    new_dir = str(write_dir/read_dir.parts[-1])
    cv2.imwrite(new_dir, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_dir, mask_to_bounding_box(Y_resized)

def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

def crop_center(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix * c / r)
    return crop(x, r_pix, c_pix, r - 2 * r_pix, c - 2 * c_pix)

def rotate_img(im, deg, rot=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c / 2, r / 2), deg, 1)
    if rot:
        return cv2.warpAffine(im, M,(c, r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def cropXYRandom(x, Y, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix * c / r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2 * rand_r * r_pix).astype(int)
    start_c = np.floor(2 * rand_c * c_pix).astype(int)
    xx = crop(x, start_r, start_c, r - 2 * r_pix, c - 2 * c_pix)
    YY = crop(Y, start_r, start_c, r - 2 * r_pix, c - 2 * c_pix)
    return xx, YY

def transforms(directory, bounding_box, transforms):
    x = cv2.imread(str(directory))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
    Y = make_mask(bounding_box, x)
    if not transforms:
        x, Y = crop_center(x), crop_center(Y)
    else:
        rdeg = (np.random.random() - 0.50) * 20
        x = rotate_img(x, rdeg)
        Y = rotate_img(Y, rdeg, rot=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = cropXYRandom(x, Y)
    return x, mask_to_bounding_box(Y)

def create_corner_rect(bounding_box, color='red'):
    bounding_box = np.array(bounding_box, dtype=np.float32)
    return plt.Rectangle((bounding_box[1], bounding_box[0]), bounding_box[3]-bounding_box[1], bounding_box[2]-bounding_box[0], color=color,
                         fill=False, lw=3)

def show_corner_bounding_box(im, bounding_box):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bounding_box))

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
        x, y_bounding_box = transforms(directory, self.bounding_box[idx], self.transforms)
        x = normalise(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bounding_box

class bounding_box_model(nn.Module):
    def __init__(self):
        super(bounding_box_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.f1 = nn.Sequential(*layers[:6])
        self.f2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 21))
        self.bounding_box = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        out = self.f1(x)
        out = self.f2(out)
        out = F.relu(out)
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = out.view(out.shape[0], -1)
        return self.classifier(out), self.bounding_box(out)
    

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
    batch_size = 64
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
        
    df_train = produce_train_df(annotation_dir)

    #label encode target
    class_dict = {'fries': 0, 'omelette': 1, 'rice': 2, 'icecream': 3, 'apple_pie': 4, 'burger': 5, 'hotdog': 6, 'beer': 7, 'bolonese_pasta': 8, 'bread': 9, 'carbonara': '10', 'carbonara_pasta': 11, 'chantilly': 12, 'coffee': 13,'cream': 14, 'fires': 15, 'ketchup': 16, 'pie': 17, 'potatoes': 18, 'salad': 19, 'sandwich': 20}
    df_train['class'] = df_train['class'].apply(lambda x:  class_dict[x])
    
    #Populating Training DF with new dirs and bounding boxes
    new_dirs = []
    new_bounding_boxs = []
    train_dir_resized = Path('./food/images_resized')
    for index, row in df_train.iterrows():
        new_dir,new_bounding_box = resize_image_bounding_box(row['filename'], train_dir_resized, generate_bounding_box_array(row.values),300)
        new_dirs.append(new_dir)
        new_bounding_boxs.append(new_bounding_box)
    df_train['new_dir'] = new_dirs
    df_train['new_bounding_box'] = new_bounding_boxs

    
    X = df_train[['new_dir', 'new_bounding_box']]
    Y = df_train['class']
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    
    train_ds = FoodDataset(X_train['new_dir'],X_train['new_bounding_box'] ,y_train, transforms=True)
    valid_ds = FoodDataset(X_val['new_dir'],X_val['new_bounding_box'],y_val)
    
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    
    model = bounding_box_model().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    train_losses, v_losses = train_epocs(model, optimizer, train_dl, valid_dl, epochs)  
    
    plot_loss_graph(epochs, train_losses, v_losses, init_folder)