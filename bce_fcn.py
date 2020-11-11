import os
import sys
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import argparse

import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, datasets, transforms
import torch.utils.data as utils

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from asymmetric import AsymmetricLossOptimized, AsymmetricLoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class ZeroPadCollator:

    @staticmethod
    def collate_tensors(batch):
        dims = batch[0].dim()
        max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
        size = (len(batch),) + tuple(max_size)
        canvas = batch[0].new_zeros(size=size)
        for i, b in enumerate(batch):
            sub_tensor = canvas[i]
            for d in range(dims):
                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
            sub_tensor.add_(b)
        return canvas

    def collate(self, batch):
        dims = len(batch[0])
        return [self.collate_tensors([b[i] for b in batch]) for i in range(dims)]

class MultiLabelDataset(Dataset):

	def __init__(self, path):

		self.path = path
		self.label_path = os.path.join('coco/labels', path.split('_')[-1])

		# Find image file names
		self.data = [f for f in tqdm.tqdm(os.listdir(self.path))] 
		self.length = len(self.data)


	def __getitem__(self, index):
		# Create image pair. Perform correlation to get target value
		feature_file = self.data[index]
		label_file = feature_file.replace('npy', 'txt')

		features = np.load(os.path.join(self.path, feature_file))
		present_classes = set()
		label_path = os.path.join(self.label_path, label_file)
		if os.path.isfile(label_path):
			labels = np.loadtxt(label_path, usecols=[0], ndmin=1)
			present_classes = set(labels.astype(int))

		label = torch.zeros(1).float()
		if 37 in present_classes:
			label = torch.ones(1).float()

		features = torch.Tensor(features)

		return features.squeeze(0), label

	def __len__(self):
		return self.length

class Net(nn.Module):
	def __init__(self):
	  super(Net, self).__init__()
	  self.conv1 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, padding_mode='reflect')
	  self.GAP =  nn.AdaptiveAvgPool2d((1,1))
	  self.fc1 = nn.Linear(2048, 80, bias=True)

	# x represents our data
	def forward(self, x):
		B, C, H, W = x.shape
		x = self.conv1(x)
		x = self.GAP(x).view(B, -1)

		x = self.fc1(x)
		return x


def eval(loader, model, criterion, optimizer, device, train=True):

	total_loss = 0
	total_acc = 0
	for data, labels in tqdm.tqdm(loader):
		data, labels = data.to(device), labels.to(device)
		optimizer.zero_grad()

		outputs = model(data)
		loss = criterion(outputs, labels)

		if train:
			loss.backward()
			optimizer.step()

		total_loss += loss.item()

	return total_loss / len(loader)


def train():
	
	try:
		shutil.rmtree('runs1')
	except:
		pass
	writer = SummaryWriter(os.path.join('runs1'))
	best_test_loss = np.inf
	
	# path = 'coco/labels/train2017'
	# pos_weights = np.ones(80)
	# if opt.scale_weights and os.path.isfile('bce_weights.npy'):
	# 	pos_weights = np.load('bce_weights.npy')
	# elif opt.scale_weights:
		
	# 	pos_weights = np.zeros(80)
	# 	for label_file in tqdm.tqdm(os.listdir(path)):

	# 		label_path = os.path.join(path, label_file)
	# 		if os.path.isfile(label_path):
	# 			labels = np.loadtxt(label_path, usecols=[0], ndmin=1)
	# 			present_classes = labels.astype(int)
	# 			for p in present_classes:
	# 				pos_weights[p] += 1

		
	# 	pos_weights /= np.max(pos_weights)
	# 	np.save('bce_weights', pos_weights)
	# print(pos_weights)

	batch_size = opt.batch_size
	num_workers = opt.num_workers
	device = torch.device('cuda:{}'.format(opt.device))
	epochs = opt.epochs

	model = models.resnet50()
	model.conv1 = torch.nn.Conv1d(1024, 64, (7, 7), (2, 2), (3, 3), bias=False)
	model.fc = nn.Linear(2048, 1)
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995, last_epoch=-1)
	# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weights).to(device))
	criterion = nn.BCEWithLogitsLoss()
	# criterion = AsymmetricLossOptimized()
	# criterion = FocalLoss(alpha=0.25, gamma=2, reduce=True)
	zero_pad = ZeroPadCollator()
	trainset = MultiLabelDataset(path='yolov3_conv_coco_train2017')
	testset = MultiLabelDataset(path='yolov3_conv_coco_val2017')
	trainloader = utils.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=zero_pad.collate)
	testloader = utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=zero_pad.collate)


	for epoch in range(epochs):
		model.train()
		train_loss = eval(trainloader, model, criterion, optimizer, device, train=True)
		print('Epoch {} | Train Loss: {}'.format(epoch, train_loss))
		writer.add_scalar('Train Loss', train_loss, epoch)
		
		with torch.no_grad():
			model.eval()
			test_loss = eval(testloader, model, criterion, optimizer, device, train=False)
			
			print('Epoch {} | Test Loss: {}'.format(epoch, test_loss))
			writer.add_scalar('Test Loss', test_loss, epoch)

			if test_loss < best_test_loss:
				torch.save(model.state_dict(), os.path.join('surfboard_fcn_best_test_weights.pt'))

		for param_group in optimizer.param_groups:
			print('learning rate: {}'.format(param_group['lr']))
		# scheduler.step()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=128, help='size of each image batch')
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--scale_weights', action='store_true', help='use loss weighting scheme')
	parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')

	opt = parser.parse_args()
	train()


