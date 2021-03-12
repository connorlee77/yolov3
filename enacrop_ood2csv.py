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

def kitti():

    frames = glob.glob(os.path.join(frames_path, '*'))

    column_names = ['image', 'class', 'x1', 'y1', 'x2', 'y2']

    classes = os.listdir(cam_path)
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
        column_names.append(c + '_cam_mean')
        column_names.append(c + '_diff_mean')
        column_names.append(c + '_grad_mean')
        column_names.append(c + '_temp_mean')

    data = []
    counter = 0
    for frame in tqdm.tqdm(frames):        

        image_name = os.path.basename(frame)
        img = cv2.imread(frame)
        H, W, C = img.shape

        matname = image_name.replace('jpg', 'mat')

        x1, x2 = 0, W
        y1, y2 = 0, H

        row = [image_name, class_name, x1,y1,x2,y2]
        for name in classes:
            cam_mat = loadmat(os.path.join(cam_path, name, matname))['cam']
            diff_mat = loadmat(os.path.join(diff_path, name, matname))['cam']
            grad_mat = loadmat(os.path.join(grad_path, name, matname))['cam']
            temp_mat = loadmat(os.path.join(temp_path, name, matname))['cam']

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

            cam_mean = np.mean(cam_bbox)
            diff_mean = np.mean(diff_bbox)
            grad_mean = np.mean(grad_bbox)
            temp_mean = np.mean(temp_bbox)
            row.append(cam_mean)
            row.append(diff_mean)
            row.append(grad_mean)
            row.append(temp_mean)


        data.append(row)

    df = pd.DataFrame(data, columns=column_names, dtype=float)
    df.to_csv(save_name)
    print(counter / len(data))
if __name__ == '__main__':
    # frames_path = '/home/fremont/ford/serengeti/ena24_crops'
    # cam_path = 'ena_crops_out_cam'
    # diff_path = 'ena_crops_out_diff'
    # temp_path = 'ena_crops_out_tempcam'
    # grad_path = 'ena_crops_out_gradient'
    # save_name = 'ena_crop_cams.csv'

    frames_path = '/home/fremont/ford/kitti/training/yolo/images_crops'
    cam_path = 'kitti_crops_out_cam'
    diff_path = 'kitti_crops_out_diff'
    temp_path = 'kitti_crops_out_tempcam'
    grad_path = 'kitti_crops_out_gradient'
    save_name = 'kitti_crop_cams.csv'
    class_name = 'kitti-crop'
    kitti()

