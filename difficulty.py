import os
import glob 
import json 
import pandas as pd 
import numpy as np
from scipy.io import loadmat 
from utils.utils import load_classes
import tqdm

JSON_FILE = os.path.join('data', 'bdd100k', 'labels', 'bdd100k_labels_images_val.json')
frames = json.load(open(JSON_FILE, 'r'))

column_names = ['image', 'class', 'x1', 'y1', 'x2', 'y2', 'height', 'width', 'area', 'occluded', 'truncated', 'weather', 'timeofday']

class_names = set(load_classes('data/coco.names'))

classes = os.listdir('bdd_out_cam')
classes.sort()
for c in classes:
    column_names.append(c + '_cam_max')
    column_names.append(c + '_diff_max')
    column_names.append(c + '_cam_mean')
    column_names.append(c + '_diff_mean')

data = []
for frame in tqdm.tqdm(frames):
    image_name = frame['name']
    weather = frame['attributes']['weather']
    timeofday = frame['attributes']['timeofday']
    labels = frame['labels']

    matname = image_name.replace('jpg', 'mat')

    for label in labels:
        class_name = label['category']

        if class_name == 'rider':
            class_name = 'person'
        
        if class_name not in class_names:
            continue

        xy = label['box2d']
        bbox = [xy['x1'], xy['y1'], xy['x2'], xy['y2']]

        x1, y1, x2, y2 = map(int, bbox)

        occluded = label['attributes']['occluded']
        truncated = label['attributes']['truncated']

        height = abs(bbox[1] - bbox[3])
        width = abs(bbox[0] - bbox[2])
        area = width*height
        row = [image_name, class_name, *bbox, height, width, area, occluded, truncated, weather, timeofday]
        
        if abs(x2 - x1) <= 0 or abs(y2 - y1) <= 0:
            print('skip')
            continue
        assert((x2 - x1)*(y2-y1) > 0)

        for name in classes:

            cam_mat = loadmat(os.path.join('bdd_out_cam', name, matname))['cam']
            diff_mat = loadmat(os.path.join('bdd_out_diff', name, matname))['cam']

            cam_bbox = cam_mat[y1:y2, x1:x2]
            diff_bbox = diff_mat[y1:y2, x1:x2]

            cam_max = np.max(cam_bbox)
            diff_max = np.max(diff_bbox)
            row.append(cam_max)
            row.append(diff_max)

            cam_mean = np.mean(cam_bbox)
            diff_mean = np.mean(diff_bbox)
            row.append(cam_mean)
            row.append(diff_mean)

        data.append(row)

df = pd.DataFrame(data, columns=column_names, dtype=float)
df.to_csv('bdd100k_val_difficulty.csv')
print(df[['height', 'width', 'area', 'occluded', 'truncated', 'timeofday']])