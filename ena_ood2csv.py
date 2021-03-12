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

classes2ignore = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bear', 'human', 'vehicle', 'American Crow', 'woodchuck', 'Chicken', 'turkey']



with open('/home/fremont/ford/serengeti/ena24.json') as f:
    data = json.load(f)

ena_classes = {}
for cat in data['categories']:
    
    keep = True
    for ignorecls in classes2ignore:
        if ignorecls.lower() in cat['name'].lower():
            keep = False
            break

    if keep:
        ena_classes[cat['id']] = cat['name']

def dataset():

    frames = glob.glob(os.path.join('/home/fremont/ford/serengeti/labels', '*.txt'))
    IMAGE_PATH = '/home/fremont/ford/serengeti/ena24_subset'

    column_names = ['image', 'class', 'x1', 'y1', 'x2', 'y2']

    classes = os.listdir('ena_out_cam')
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
        labels = np.genfromtxt(frame, delimiter=" ", dtype=np.float)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)
        image_name = os.path.basename(frame).replace('txt', 'jpg')
        img = cv2.imread(os.path.join(IMAGE_PATH, image_name))
        H, W, C = img.shape

        matname = image_name.replace('jpg', 'mat')
        for label in labels:
            class_num = int(label[0])
            class_name = ena_classes[class_num - 80]

            bbox = label[1:].astype(float)
            bbox[[0,2]] *= W
            bbox[[1,3]] *= H

            x1, x2 = int(bbox[0] - bbox[2]/2), int(bbox[0] + bbox[2]/2)
            y1, y2 = int(bbox[1] - bbox[3]/2), int(bbox[1] + bbox[3]/2)

            row = [image_name, class_name, x1,y1,x2,y2]
            for name in classes:
                cam_mat = loadmat(os.path.join('ena_out_cam', name, matname))['cam']
                diff_mat = loadmat(os.path.join('ena_out_diff', name, matname))['cam']
                grad_mat = loadmat(os.path.join('ena_out_gradient', name, matname))['cam']
                temp_mat = loadmat(os.path.join('ena_out_tempcam', name, matname))['cam']

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
    df.to_csv('ena_cams.csv')
    print(counter / len(data))
if __name__ == '__main__':
    
    dataset()

