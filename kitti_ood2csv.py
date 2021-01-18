import os
import glob 
import json 
import pandas as pd 
import numpy as np
from scipy.io import loadmat 
from utils.utils import load_classes
import tqdm
import cv2
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

with open('data/coco.names') as f:
    coco_names = [x.strip() for x in f.readlines()]

def kitti(setname):

    kitti2coco = {
        'Car' : 'car',
        'Van' : 'car',
        'Truck' : 'truck',
        'Pedestrian' : 'person',
        'Person_sitting' : 'person',
        'Cyclist' : 'person',
        'Tram' : 'train'
    }

    frames = glob.glob(os.path.join('/home/fremont/ford/kitti/training/yolo/kitti_ood/labels', '*.txt'))
    IMAGE_PATH = '/home/fremont/ford/kitti/training/yolo/kitti_ood/images'

    column_names = ['image', 'class', 'x1', 'y1', 'x2', 'y2']

    class_names = set(load_classes('data/coco.names'))

    classes = os.listdir('kitti_{}_out_cam'.format(setname))
    classes.sort()
    for c in classes:
        column_names.append(c + '_cam_max')
        column_names.append(c + '_diff_max')
        column_names.append(c + '_grad_max')
        column_names.append(c + '_ratio_max')
        column_names.append(c + '_cam_mean')
        column_names.append(c + '_diff_mean')
        column_names.append(c + '_grad_mean')
        column_names.append(c + '_ratio_mean')

    data = []
    counter = 0
    for frame in tqdm.tqdm(frames):        
        labels = np.genfromtxt(frame, delimiter=" ", dtype=np.float)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)
        image_name = os.path.basename(frame).replace('txt', 'png')
        img = cv2.imread(os.path.join(IMAGE_PATH, image_name))
        H, W, C = img.shape

        matname = image_name.replace('png', 'mat')
        for label in labels:
            class_num = int(label[0])
            if class_num >= 0:
                class_name = coco_names[class_num]
            else:
                class_name = 'ood-mnist'

            bbox = label[1:].astype(float)
            bbox[[0,2]] *= W
            bbox[[1,3]] *= H

            x1, x2 = int(bbox[0] - bbox[2]/2), int(bbox[0] + bbox[2]/2)
            y1, y2 = int(bbox[1] - bbox[3]/2), int(bbox[1] + bbox[3]/2)


            row = [image_name, class_name, x1,y1,x2,y2]
            for name in classes:
                cam_mat = loadmat(os.path.join('kitti_{}_out_cam'.format(setname), name, matname))['cam']
                diff_mat = loadmat(os.path.join('kitti_{}_out_diff'.format(setname), name, matname))['cam']
                grad_mat = loadmat(os.path.join('kitti_{}_out_gradient'.format(setname), name, matname))['cam']
                ratio_mat = loadmat(os.path.join('kitti_{}_out_ratio'.format(setname), name, matname))['cam']

                cam_bbox = cam_mat[y1:y2, x1:x2]
                diff_bbox = diff_mat[y1:y2, x1:x2]
                grad_bbox = grad_mat[y1:y2, x1:x2]
                ratio_bbox = ratio_mat[y1:y2, x1:x2]

                cam_max = np.max(cam_bbox)
                diff_max = np.max(diff_bbox)
                grad_max = np.max(grad_bbox)
                ratio_max = np.max(ratio_bbox)
                row.append(cam_max)
                row.append(diff_max)
                row.append(grad_max)
                row.append(ratio_max)

                cam_mean = np.mean(cam_bbox)
                diff_mean = np.mean(diff_bbox)
                grad_mean = np.mean(grad_bbox)
                ratio_mean = np.mean(ratio_bbox)
                row.append(cam_mean)
                row.append(diff_mean)
                row.append(grad_mean)
                row.append(ratio_mean)

            data.append(row)
    df = pd.DataFrame(data, columns=column_names, dtype=float)
    df.to_csv('kitti_{}01.csv'.format(setname))
    print(counter / len(data))
if __name__ == '__main__':
    
    kitti('ood')

