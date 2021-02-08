import cv2
import numpy as np 
from scipy.io import loadmat
import glob 
import os
import tqdm

def normalize(img):
	img = img - np.min(img)
	img = img / np.max(img)
	return img

KITTI_CAM_PATH = 'kitti_ood_1eN4_noclip_out_cam'
KITTI_GRAD_PATH = 'kitti_ood_1eN4_noclip_out_gradient'
# KITTI_IMAGE_PATH = '/home/fremont/ford/kitti/training/yolo/images'
KITTI_IMAGE_PATH = 'kitti_ood_stat_samples'
# KITTI_IMAGE_PATH = '/home/fremont/ford/kitti/training/yolo/kitti_ood/images'


images = glob.glob(os.path.join(KITTI_IMAGE_PATH, '*'))


grad_lst = []
cam_lst = []
for image_path in tqdm.tqdm(images):
	img_name = os.path.basename(image_path).split('.')[0]
	img_name = '000349'
	image_path = 'kitti_ood_stat_samples/000349.png'
	img = cv2.imread(image_path)
	print(img.shape)
	cam = loadmat(os.path.join(KITTI_CAM_PATH, 'car', '{}.mat'.format(img_name)))['cam']
	grad = loadmat(os.path.join(KITTI_GRAD_PATH, 'car', '{}.mat'.format(img_name)))['cam']

	grad_lst.append(np.percentile(grad, 99))

	grad[grad > np.percentile(grad, 99)] = np.percentile(grad, 99)
	# grad[grad > 0.15] = 0.15
	grad = normalize(grad)
	grad_mask = grad < 0.333
	grad = cv2.applyColorMap(np.uint8(255*grad), cv2.COLORMAP_JET)
	grad[grad_mask,:] = img[grad_mask,:]

	cam_lst.append(np.max(cam))
	cam = normalize(cam)
	cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

	overlay_cam = img*0.5 + 0.4*cam
	overlay_grad = img*0.5 + 0.5*grad
	# cv2.imwrite(os.path.join('kitti_cams_vis', '{}_cam.jpg'.format(img_name)), overlay_cam)
	# cv2.imwrite(os.path.join('kitti_grads_vis', '{}_grad.jpg'.format(img_name)), overlay_grad)
	cv2.imshow(img_name, np.uint8(overlay_grad))
	cv2.imshow(img_name, np.uint8(overlay_cam))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


cam_lst = np.array(cam_lst)
print(np.percentile(cam_lst, 10))
grad_lst = np.array(grad_lst)
print(np.percentile(grad_lst, 10))