import sys
import os
import shutil

train_images = []
with open('data/VOC2012/ImageSets/Main/train.txt') as f:
		for i in f:
			train_images.append(i[:-1])
val_images = []
with open('data/VOC2012/ImageSets/Main/val.txt') as f:
	for i in f:
		val_images.append(i[:-1])

for i in os.listdir('data/VOC2012/labels'):
	file = os.path.join('data/VOC2012/labels/', i)
	if os.path.isfile(file):
		if i.split('.')[0] in train_images:
			shutil.move(file, 'data/VOC2012/labels/train/' + i)
		elif i.split('.')[0] in val_images:
			shutil.move(file, 'data/VOC2012/labels/val/' + i)