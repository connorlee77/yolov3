import os
import sys
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, datasets, transforms
import torch.utils.data as utils

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

class MultiLabelDataset(Dataset):

	def __init__(self, path):

		self.path = path
		self.label_path = os.path.join('coco/labels', path.split('_')[2])

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

		label = np.zeros(80)
		label[list(present_classes)] = 1

		return features.flatten(), label

	def __len__(self):
		return self.length

class Net(nn.Module):
	def __init__(self):
	  super(Net, self).__init__()
	  self.fc1 = nn.Linear(1024, 80)

	# x represents our data
	def forward(self, x):
	  x = self.fc1(x)
	  return x


def eval(loader, train=True):

	total_loss = 0
	total_acc = 0
	for data, labels in tqdm.tqdm(loader):
		data, labels = data.to(device), labels.to(device)

		outputs = model(data)
		loss = criterion(outputs, labels)

		outputs = torch.sigmoid(outputs)

		preds = outputs > threshold
		matches = labels.int() == preds.int()
		accuracy = torch.mean(matches.sum(dim=1).float() / 80)

		total_loss += loss.item()
		total_acc += accuracy.item()

	return total_loss / len(loader), total_acc / len(loader)


def test():
	with torch.no_grad():
		model.eval()
		test_loss, test_acc = eval(testloader, train=False)
		print(test_loss, test_acc)
if __name__ == '__main__':
	# dataset = MultiLabelDataset(path='coco_train2017')
	# loader = utils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
	# for i in tqdm.tqdm(range(dataset.__len__())):
	# 	dataset.__getitem__(i)

	batch_size = 512
	num_workers = 8
	device = torch.device('cuda:0')
	threshold = 0.8

	model = Net()
	model.load_state_dict(torch.load('fcn_best_test_weights.pt'))
	model.to(device)
	criterion = nn.BCEWithLogitsLoss()
	
	testset = MultiLabelDataset(path='yolov3_coco_val2017')
	testloader = utils.DataLoader(testset, batch_size=512, shuffle=False, num_workers=1, pin_memory=True)
	test()

