#library imports
import os
import pandas as pd
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET

def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def produce_train_df(annotation_dir, images_dir):
    annotations = filelist(annotation_dir, '.xml')
    anno_list = []
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

        ## Convert labels 
        converted_anno = convert_labels(str(annotation['filename']), annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax'])
        annotation['x'], annotation['y'], annotation['w'], annotation['h'] = converted_anno

        anno_list.append(annotation)
    return pd.DataFrame(anno_list)

def sorting(l1, l2):
    if l1 > l2:
        lmax, lmin = l1, l2
        return lmax, lmin
    else:
        lmax, lmin = l2, l1
        return lmax, lmin

def convert_labels(path, x1, y1, x2, y2):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.
        This converts (x1, y1, x1, y2) KITTI format to
        (x_centre, y_centre, width, height) normalized YOLO format to be used for darknet.
    """

    size = get_img_shape(path)
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x_centre = ((xmin + xmax)/2.0) * dw
    y_centre = ((ymin + ymax)/2.0) * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh

    return (x_centre,y_centre,w,h)

def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print('error, could not find this path: ', path)
        return (None, None, None)

if __name__ == '__main__':
    img_path = 'data/custom/images/'
    anno_path = 'data/custom/annotations/'
    images_dir = Path(f'./{img_path}')
    annotation_dir = Path(f'./{anno_path}')
    perc = 70 # Percentage of training data for train dataset
    labels_dir = 'data/custom/labels/'
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    df_yolo = produce_train_df(annotation_dir, images_dir)
    class_dict = {'fries': 0, 'omelette': 1, 'rice': 2, 'icecream': 3, 'apple_pie': 4, 'burger': 5, 'hotdog': 6,
                  'beer': 7, 'bolonese_pasta': 8, 'bread': 9, 'carbonara': '10', 'carbonara_pasta': 11, 'chantilly': 12,
                  'coffee': 13, 'cream': 14, 'fires': 15, 'ketchup': 16, 'pie': 17, 'potatoes': 18, 'salad': 19,
                  'sandwich': 20}
    df_yolo['class'] = df_yolo['class'].apply(lambda x: class_dict[x])

    # Generate annotations for each image in darknet format
    for i in range(len(df_yolo)):
        file_name = str(df_yolo['filename'][i]).replace(img_path, '').replace('.jpg', '')
        annotation = f"{df_yolo['class'][i]} {df_yolo['x'][i]} {df_yolo['y'][i]} {df_yolo['w'][i]} {df_yolo['h'][i]}"
        text_file = open(f"{labels_dir}{file_name}.txt", "w")
        n = text_file.write(annotation)
        text_file.close()

    # Generate class.names file
    names_file = open("data/custom/class.names", "w")
    for x in list(class_dict.keys()):
        n = names_file.write(f"{x}\n")
    names_file.close()

    # Create the train and validation files
    train_file = open("data/custom/train.txt", "w")
    valid_file = open("data/custom/valid.txt", "w")

    tot_len = len(df_yolo)
    split = int(perc/100 * tot_len)

    for i in range(tot_len):
      if i < split:
        n = train_file.write(f"{str(df_yolo['filename'][i])}\n")
      else:
        n = valid_file.write(f"{str(df_yolo['filename'][i])}\n")

    train_file.close()
    valid_file.close()



