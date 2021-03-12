import cv2
import numpy as np 
from scipy.io import loadmat
import glob 
import os
import tqdm

def normalize(img, mini, maxi):
	x = (img - mini) / (maxi - mini)
	return x

KITTI_IMAGE_PATH = '/home/fremont/ford/kitti/training/yolo/images'

images = glob.glob(os.path.join(KITTI_IMAGE_PATH, '*'))
image_dir = ['images', '4000', '7000', '10000']


for i, image_path in tqdm.tqdm(enumerate(images)):
	img_name = os.path.basename(image_path).split('.')[0]
	
	minCam, maxCam = 1000000, -1
	minGrad, maxGrad = 100000000, -1
	for exp in ['', '4k_', '7k_', '10k_']:
		cam = loadmat(os.path.join('kitti_{}out_cam'.format(exp), 'car', '{}.mat'.format(img_name)))['cam']
		grad = loadmat(os.path.join('kitti_{}out_gradient'.format(exp), 'car', '{}.mat'.format(img_name)))['cam']

		print(exp, np.percentile(grad, 99), np.max(grad))

		minCam = min(minCam, np.min(cam))
		maxCam = max(maxCam, np.max(cam))

		minGrad = min(minGrad, np.min(grad))
		maxGrad = max(maxGrad, np.percentile(grad, 99))

	# continue
	assert(minCam < 100 and maxCam >= 0 and minGrad < 100 and maxGrad >= 0)
	print(minGrad, maxGrad)
	for k, exp in enumerate(['', '4k_', '7k_', '10k_']):
		setname = image_dir[k]
		image_path = os.path.join('/home/fremont/ford/kitti/training/yolo/{}'.format(setname), os.path.basename(image_path))
		img = cv2.imread(image_path)

		cam = loadmat(os.path.join('kitti_{}out_cam'.format(exp), 'car', '{}.mat'.format(img_name)))['cam']
		grad = loadmat(os.path.join('kitti_{}out_gradient'.format(exp), 'car', '{}.mat'.format(img_name)))['cam']

		# grad[grad > np.percentile(grad, 99)] = np.percentile(grad, 99)
		grad = normalize(grad, minGrad, maxGrad)
		# grad_mask = grad < 0.333
		grad = cv2.applyColorMap(np.uint8(255*grad), cv2.COLORMAP_JET)
		# grad[grad_mask,:] = img[grad_mask,:]

		cam = normalize(cam, minCam, maxCam)
		cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

		overlay_cam = img*0.5 + 0.4*cam
		overlay_grad = img*0.5 + 0.5*grad
		cv2.imwrite(os.path.join('kitti_cams_vis{}'.format(exp.replace('_', '')), '{}_cam.jpg'.format(img_name)), overlay_cam)
		cv2.imwrite(os.path.join('kitti_grads_vis{}'.format(exp.replace('_', '')), '{}_grad.jpg'.format(img_name)), overlay_grad)

	if i == 200:
		break