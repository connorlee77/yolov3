import os
import glob
import numpy as np 
import pandas as pd 
import cv2
import tqdm
from scipy.io import loadmat 
from utils.utils import load_classes

def make_folder(out):
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

setname = 'ood'
num_name = 'kitti_ood/images'
SAVE_IMAGE_DIR = 'kitti_{}_stat_samples'.format(setname)
make_folder(SAVE_IMAGE_DIR)


# IMAGE_DIR = 'data/bdd100k/images/100k/val'
# PREDICTION_PATH = 'predlabeled'
# classes = os.listdir('bdd_out_cam')

# IMAGE_DIR = '/home/fremont/ford/kitti/training/yolo/images'
IMAGE_DIR = '/home/fremont/ford/kitti/training/yolo/{}'.format(num_name)

PREDICTION_PATH = 'kitti_{}_predlabeled'.format(setname)
classes = os.listdir('kitti_{}_out_cam'.format(setname))

classes.sort()

pred_files = glob.glob(os.path.join(PREDICTION_PATH, '*'))
class_names = load_classes('data/coco.names')

type_bbox = ['FN', 'FP', 'TP']

data = []
column_names = ['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class', 'type']
for c in classes:
	column_names.append(c + '_cam_max')
	column_names.append(c + '_diff_max')
	column_names.append(c + '_cam_mean')
	column_names.append(c + '_diff_mean')

for it, obj_inst in tqdm.tqdm(enumerate(pred_files), total=len(pred_files)):

	filename = os.path.basename(obj_inst)
	# imagename = filename.replace('npy', 'jpg')
	imagename = filename.replace('npy', 'png')
	matname = filename.replace('npy', 'mat')

	pred = np.load(obj_inst).astype(np.float)
	if len(pred) == 0:
		continue

	pred_bboxes = pred[:,0:4].astype(np.int)

	if it % 1 == 0:
		image = cv2.imread(os.path.join(IMAGE_DIR, imagename))
		for i, p in enumerate(pred_bboxes):
			if pred[i,6] == 1:
				color = (0, 255, 0)
			elif pred[i,6] == 0:
				color = (0, 0, 255)
			else:
				color = (255, 0, 0)

			cv2.rectangle(image, (p[0], p[1]), (p[2], p[3]), color, 1)
			cv2.putText(image, '{} | {} | {:.2f}'.format(type_bbox[int(pred[i,6]) + 1], class_names[int(pred[i,5])], pred[i,4]), (p[0], p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

		cv2.imwrite(os.path.join(SAVE_IMAGE_DIR, imagename), image)

	for i, p in enumerate(pred_bboxes):
		x1, y1, x2, y2 = p 

		if abs(x2 - x1) <= 0 or abs(y2 - y1) <= 0:
			continue
		assert((x2 - x1)*(y2-y1) > 0)

		row = [filename.split('.')[0], *p, pred[i,4], class_names[int(pred[i,5])], type_bbox[int(pred[i,6]) + 1]]

		for name in classes:

			cam_mat = loadmat(os.path.join('kitti_{}_out_cam'.format(setname), name, matname))['cam']
			diff_mat = loadmat(os.path.join('kitti_{}_out_diff'.format(setname), name, matname))['cam']

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
df.to_csv('kitti_{}_cams.csv'.format(setname))
print(df)