import cv2
import numpy as np 
from scipy.io import loadmat
import glob 
import os

def normalize(img):
	img = img - np.min(img)
	img = img / np.max(img)
	return img

KITTI_CAM_PATH = 'kitti_out_cam'
KITTI_GRAD_PATH = 'kitti_out_gradient'
# KITTI_IMAGE_PATH = '/home/fremont/ford/kitti/training/yolo/images'
KITTI_IMAGE_PATH = 'kitti_stat_samples'

images = glob.glob(os.path.join(KITTI_IMAGE_PATH, '*'))

for image_path in images:
	img_name = os.path.basename(image_path).split('.')[0]
	print(img_name)
	img = cv2.imread(image_path)
	cam = loadmat(os.path.join(KITTI_CAM_PATH, 'car', '{}.mat'.format(img_name)))['cam']
	grad = loadmat(os.path.join(KITTI_GRAD_PATH, 'car', '{}.mat'.format(img_name)))['cam']

	# grad[grad > np.percentile(grad, 99)] = np.percentile(grad, 99)
	grad = normalize(grad)
	grad = cv2.applyColorMap(np.uint8(255*grad), cv2.COLORMAP_JET)

	cam = normalize(cam)
	cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

	# cv2.imshow('cam', cam)
	# cv2.waitKey(0)

	overlay_img = img*0.5 + 0.3*grad
	cv2.imshow('img', np.uint8(overlay_img))
	cv2.waitKey(0)
