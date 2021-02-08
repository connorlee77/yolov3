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

    frames = glob.glob(os.path.join('/home/fremont/ford/serengeti/ena24_crop/Virginia_Opossum', '*.jpg'))

    column_names = ['image', 'class', 'x1', 'y1', 'x2', 'y2']
    class_names = set(load_classes('data/coco.names'))

    classes = os.listdir('ena24_opposum_cam')
    classes.sort()
    for c in classes:
        column_names.append(c + '_cam_max')
        column_names.append(c + '_diff_max')
        column_names.append(c + '_grad_max')
        column_names.append(c + '_temp_max')
        column_names.append(c + '_cam_median')
        column_names.append(c + '_diff_median')
        column_names.append(c + '_grad_median')
        column_names.append(c + '_temp_median')

    data = []
    counter = 0
    for frame in tqdm.tqdm(frames):        

        image_name = os.path.basename(frame)
        img = cv2.imread(frame)
        H, W, C = img.shape

        matname = image_name.replace('jpg', 'mat')

        class_name = 'ood'

        x1, x2 = 0, W
        y1, y2 = 0, H

        row = [image_name, class_name, x1,y1,x2,y2]
        for name in classes:
            cam_mat = loadmat(os.path.join('ena24_opposum_cam', name, matname))['cam']
            diff_mat = loadmat(os.path.join('ena24_opposum_diff', name, matname))['cam']
            grad_mat = loadmat(os.path.join('ena24_opposum_gradient', name, matname))['cam']
            temp_mat = loadmat(os.path.join('ena24_opposum_tempcam', name, matname))['cam']

            cam_bbox = cam_mat[y1:y2, x1:x2]
            diff_bbox = diff_mat[y1:y2, x1:x2]
            grad_bbox = grad_mat[y1:y2, x1:x2]
            temp_bbox = temp_mat[y1:y2, x1:x2]

            cam_max = np.max(cam_bbox)
            diff_max = np.max(diff_bbox)
            grad_max = np.max(grad_bbox)
            temp_max = np.max(temp_bbox)
            row.append(cam_max)
            row.append(diff_max)
            row.append(grad_max)
            row.append(temp_max)

            cam_median = np.percentile(cam_bbox, 50)
            diff_median = np.percentile(diff_bbox, 50)
            grad_median = np.percentile(grad_bbox, 50)
            temp_median = np.percentile(temp_bbox, 50)
            row.append(cam_median)
            row.append(diff_median)
            row.append(grad_median)
            row.append(temp_median)

        data.append(row)

    df = pd.DataFrame(data, columns=column_names, dtype=float)
    df.to_csv('ena24crop_opposum.csv'.format(setname))
    print(counter / len(data))
if __name__ == '__main__':
    
    kitti('ood')

