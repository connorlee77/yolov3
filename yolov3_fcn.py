import os
import sys
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import argparse
import math
import random

import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, datasets, transforms
import torch.utils.data as utils
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from asymmetric import AsymmetricLossOptimized, AsymmetricLoss
from utils.multilabel_datasets import LoadImagesAndLabels

from models import Darknet, attempt_download

hyp = {'giou': 3.54,  # giou loss gain
	   'cls': 37.4,  # cls loss gain
	   'cls_pw': 1.0,  # cls BCELoss positive_weight
	   'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
	   'obj_pw': 1.0,  # obj BCELoss positive_weight
	   'iou_t': 0.20,  # iou training threshold
	   'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
	   'lrf': 0.0005,  # final learning rate (with cos scheduler)
	   'momentum': 0.937,  # SGD momentum
	   'weight_decay': 0.0005,  # optimizer weight decay
	   'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
	   'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
	   'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
	   'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
	   'degrees': 1.98 * 0,  # image rotation (+/- deg)
	   'translate': 0.05 * 0,  # image translation (+/- fraction)
	   'scale': 0.05 * 0,  # image scale (+/- gain)
	   'shear': 0.641 * 0}  # image shear (+/- deg)



class FocalLoss(nn.Module):
	def __init__(self, weights=1, gamma=2, logits=True, reduce=True):
		super(FocalLoss, self).__init__()
		self.weights = weights
		self.gamma = gamma
		self.logits = logits
		self.reduce = reduce

	def forward(self, inputs, targets):
		if self.logits:
			BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
		else:
			BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
		pt = torch.exp(-BCE_loss)
		F_loss = (1-pt)**self.gamma * BCE_loss

		if self.reduce:
			return torch.mean(F_loss)
		else:
			return F_loss

# class FocalLoss(nn.Module):
#     def __init__(self, weights=0.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = weights
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         p = torch.sigmoid(inputs)
#         a = self.alpha
#         gamma = self.gamma

#         loss = -a*(1-p)**gamma*torch.log(torch.clamp(p, min=1e-8))*targets - (1-a)*p**gamma*torch.log(torch.clamp(1-p, min=1e-8))*(1-targets)
#         return loss.mean()


class Net(nn.Module):
	def __init__(self, base_model):
		super(Net, self).__init__()
		# self.conv1 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, padding_mode='reflect')
		# self.GAP =  nn.AdaptiveAvgPool2d((1,1))

		self.activations = {}
		def get_activation(layer):
			def hook(module, input, output):
				self.activations[layer] = output
			return hook

		self.base_model = base_model

		for target_layer in [80, 92, 104]:
			layer = self.base_model.module_list[target_layer]
			layer.register_forward_hook(get_activation(target_layer))

		self.GAP =  nn.AdaptiveAvgPool2d((1,1))
		self.fc1 = nn.Linear(1024 + 512 + 256, 1024)
		self.fc2 = nn.Linear(1024, 80)

	# x represents our data
	def forward(self, x):
		B, C, H, W = x.shape
		
		sig_pred, pred, _ = self.base_model(x)

		act_80 = self.activations[80]
		act_92 = self.activations[92]
		act_104 = self.activations[104]
		
		act_80 = F.upsample(act_80, size=act_104.shape[2:4], mode='bilinear', align_corners=False)
		act_92 = F.upsample(act_92, size=act_104.shape[2:4], mode='bilinear', align_corners=False)

		x = torch.cat([act_80, act_92, act_104], dim=1)
		x = self.GAP(x)

		x = self.fc1(x.view(B,-1))
		x = F.relu(x)
		x = self.fc2(x)
		return x

def eval(loader, model, criterion, optimizer, device, train=True):

	total_loss = 0
	total_acc = 0
	pbar = tqdm.tqdm(loader)
	for data in pbar:		
		imgs, labels, path, shapes = data[0].to(device), data[1].to(device), data[2], data[3]
		imgs = imgs.float() / 255

		optimizer.zero_grad()

		if train and opt.multi_scale:
			img_size = random.randrange(grid_min, grid_max + 1) * gs
			sf = img_size / max(imgs.shape[2:])  # scale factor
			ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
			imgs = F.upsample(imgs, size=ns, mode='bilinear', align_corners=False)
		
		outputs = model(imgs)
		loss = criterion(outputs, labels)

		if train:
			loss.backward()
			optimizer.step()

		preds = torch.sigmoid(outputs) > 0.8
		matches = labels.int() == preds.int()
		accuracy = torch.mean(matches.sum(dim=1).float() / 80)

		total_loss += loss.item()
		total_acc += accuracy.item()

		pbar.set_description('loss: {}'.format(loss.item()))

	return total_loss / len(loader), total_acc / len(loader)


def train():
	
	writer = SummaryWriter(os.path.join('fcn_runs', opt.exp_name))
	best_test_loss = np.inf
	
	batch_size = opt.batch_size
	num_workers = opt.num_workers
	device = torch.device('cuda:{}'.format(opt.device))
	epochs = opt.epochs

	base_model = Darknet(opt.cfg)
	# Load weights
	weights = 'weights/yolov3.pt'
	attempt_download(weights)
	if weights.endswith('.pt'):  # pytorch format
		base_model.load_state_dict(torch.load(weights, map_location=device)['model'])
	else:  # darknet format
		load_darknet_weights(base_model, weights)

	model = Net(base_model)
	model.to(device)

	for module in model.base_model.module_list:
		for parameter in module.parameters():
			parameter.requires_grad = False

	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995, last_epoch=-1)
	criterion = FocalLoss(weights=None, gamma=2)
	
	train_path = 'coco/train2017.txt'
	test_path = 'coco/val2017.txt'
	trainset = LoadImagesAndLabels(train_path, img_size, batch_size, augment=True, hyp=hyp, rect=opt.rect, cache_images=False, single_cls=False)
	testset = LoadImagesAndLabels(test_path, imgsz_test, batch_size, hyp=hyp, rect=True, cache_images=True, single_cls=False)

	# trainloader
	nw = num_workers
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=nw,
											 shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
											 pin_memory=True, collate_fn=trainset.collate_fn)
	# Testloader
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=nw, pin_memory=True, collate_fn=trainset.collate_fn)


	for epoch in range(epochs):
		model.train()

		train_loss, train_acc = eval(trainloader, model, criterion, optimizer, device, train=True)
		print('Epoch {} | Train Loss: {}'.format(epoch, train_loss))
		writer.add_scalar('Train Loss', train_loss, epoch)
		
		with torch.no_grad():
			model.eval()
			test_loss, test_acc = eval(testloader, model, criterion, optimizer, device, train=False)
			
			print('Epoch {} | Test Loss: {}'.format(epoch, test_loss))
			writer.add_scalar('Test Loss', test_loss, epoch)

			if test_loss < best_test_loss:
				torch.save(model.state_dict(), os.path.join('{}_test_weights.pt'.format(opt.exp_name)))
				best_test_loss = test_loss

		for param_group in optimizer.param_groups:
			print('learning rate: {}'.format(param_group['lr']))
		scheduler.step()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
	parser.add_argument('--exp_name', type=str, default='exp001', help='*.cfg path')

	### Dont change ###
	parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
	parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
	parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
	parser.add_argument('--rect', action='store_true', help='rectangular training')

	opt = parser.parse_args()

	opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
	imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

	# Image Sizes
	gs = 32  # (pixels) grid size
	assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
	opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
	if opt.multi_scale:
		if imgsz_min == imgsz_max:
			imgsz_min //= 1.5
			imgsz_max //= 0.667
		grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
		imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
	img_size = imgsz_max  # initialize with max size
	

	train()


