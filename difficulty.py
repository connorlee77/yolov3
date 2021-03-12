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

def bdd():
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

    DEPTH_DATA_PATH = '/media/hdd2/ford/kitti_object_depth/vec/data_object_image_2/training/image_2'
    frames = glob.glob(os.path.join('/home/fremont/ford/kitti/training/label_2', '*.txt'))
    gradient_PATH = 'kitti_{}out_gradient'.format(setname)

    KITTI_IMAGE_PATH = '/home/fremont/ford/kitti/training/yolo/images'
    column_names = ['image', 'class', 'x1', 'y1', 'x2', 'y2', 'height', 'width', 'area', 'truncated', 'occluded', 'alpha', 'difficulty', 'depth_max', 'depth_min', 'depth_mean']

    class_names = set(load_classes('data/coco.names'))

    classes = os.listdir('kitti_{}out_cam'.format(setname))
    classes.sort()
    for c in classes:
        column_names.append(c + '_cam_max')
        column_names.append(c + '_diff_max')
        column_names.append(c + '_grad_max')
        column_names.append(c + '_cam_mean')
        column_names.append(c + '_diff_mean')
        column_names.append(c + '_grad_mean')

    data = []
    counter = 0
    for frame in tqdm.tqdm(frames):
        frame = '/home/fremont/ford/kitti/training/label_2/000002.txt'        
        labels = np.genfromtxt(frame, delimiter=" ", dtype='str')
        image_name = os.path.basename(frame).replace('txt', 'png')

        
        img = cv2.imread(os.path.join(KITTI_IMAGE_PATH, image_name))
        # cv2.imshow('imig', img)
        # cv2.waitKey(0)

        depth_image_path = os.path.join(DEPTH_DATA_PATH, image_name)
        zimg = 99505.81485 / np.array(Image.open(depth_image_path))
        # zimg = gaussian_filter(zimg, sigma=2)
        import matplotlib.pyplot as plt
        zimg[zimg > np.percentile(zimg, 99.5)] = np.percentile(zimg, 99.5)
        plt.imshow(zimg)
        plt.show()
        exit(0)
        matname = image_name.replace('png', 'mat')
        for label in labels:
            kitti_name = label[0]
            
            if kitti_name not in kitti2coco:
                continue

            class_name = kitti2coco[kitti_name]

            truncated = float(label[1])
            occluded = float(label[2])
            alpha = float(label[3])
            bbox = label[4:8].astype(float)

            x1, y1, x2, y2 = map(int, bbox)

            height = abs(bbox[1] - bbox[3])
            width = abs(bbox[0] - bbox[2])
            depth = zimg[y1:y2, x1:x2]
            depth_max, depth_min, depth_mean = np.max(depth), np.min(depth), np.mean(depth)
            area = width*height
            
            if height >= 40 and (occluded in [0]) and truncated <= 0.15:
                difficulty = 'easy'
            elif height >= 25 and (occluded in [0, 1]) and truncated <= 0.3:
                difficulty = 'medium'
            elif height >= 25 and (occluded in [0, 1, 2]) and truncated <= 0.5:
                difficulty = 'hard'
            else:
                difficulty = 'extreme/unkown'
                counter += 1

            row = [image_name, class_name, *bbox, height, width, area, truncated, occluded, alpha, difficulty, depth_max, depth_min, depth_mean]
            
            if abs(x2 - x1) <= 0 or abs(y2 - y1) <= 0:
                print('skip')
                continue
            assert((x2 - x1)*(y2-y1) > 0)

            for name in classes:
                cam_mat = loadmat(os.path.join('kitti_{}out_cam'.format(setname), name, matname))['cam']
                diff_mat = loadmat(os.path.join('kitti_{}out_diff'.format(setname), name, matname))['cam']
                temp_mat = loadmat(os.path.join('kitti_{}out_gradient'.format(setname), name, matname))['cam']

                cam_bbox = cam_mat[y1:y2, x1:x2]
                diff_bbox = diff_mat[y1:y2, x1:x2]
                temp_bbox = temp_mat[y1:y2, x1:x2]

                cam_max = np.max(cam_bbox)
                diff_max = np.max(diff_bbox)
                temp_max = np.max(temp_bbox)
                row.append(cam_max)
                row.append(diff_max)
                row.append(temp_max)

                cam_mean = np.mean(cam_bbox)
                diff_mean = np.mean(diff_bbox)
                temp_mean = np.mean(temp_bbox)
                row.append(cam_mean)
                row.append(diff_mean)
                row.append(temp_mean)

            data.append(row)
    df = pd.DataFrame(data, columns=column_names, dtype=float)
    # df.to_csv('kitti_{}val_difficulty_v2.csv'.format(setname))
    print(df[['height', 'width', 'area', 'occluded', 'truncated', 'alpha', 'difficulty']])
    print(counter / len(data))
if __name__ == '__main__':
    
    # kitti('ood')

    for setname in ['', '4k_', '6k_', '7k_', '10k_']:
        kitti(setname)

