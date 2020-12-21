import sys
import os
import json
import argparse
from tqdm import tqdm
from xml.dom import minidom

import xml.etree.ElementTree as ET 

#1280x720
def conversion():
	names = []
	train_images = []
	val_images = []
	writes_train = []
	writes_val = []
	if not os.path.exists('out'):
		os.makedirs('out') 
	with open('data/coco.names') as f:
		for i in f:
			names.append(i[:-1])
	with open('data/VOC2012/ImageSets/Main/train.txt') as f:
		for i in f:
			train_images.append(i[:-1])
	with open('data/VOC2012/ImageSets/Main/val.txt') as f:
		for i in f:
			val_images.append(i[:-1])
	for count, i in tqdm(enumerate(os.listdir('data/VOC2012/Annotations'))):
		root = ET.parse('data/VOC2012/Annotations/' + i).getroot()
		filename = root.find('filename').text.split('.')[0]+'.txt'
		size = root.find('size')
		picwidth = int(size.find('width').text)
		picheight = int(size.find('height').text)
		with open('out/' + filename, 'w') as f:
			for obj in root.iter('object'):
				xmin = float(obj.find('bndbox/xmin').text)
				ymin = float(obj.find('bndbox/ymin').text)
				xmax = float(obj.find('bndbox/xmax').text)
				ymax = float(obj.find('bndbox/ymax').text)
				name = obj.find('name').text
				name = name.lower()
				if(name == 'motorbike'):
					name = 'motorcycle'
				if(name == 'aeroplane'):
					name = 'airplane'
				if(name == 'tv/monitor'):
					name = 'tv'
				if(name == 'sofa'):
					name = 'couch'
				if(name not in names):
					continue
				name = names.index(name)
				width = (xmax - xmin)/picwidth
				height = (ymax - ymin)/picheight
				center_x = xmin/picwidth + width/2
				center_y = ymin/picheight + height/2
				f.write(str(name) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height) + '\n')
			if(filename.split('.')[0] in train_images):
				writes_train.append('./images/train/' + filename.split('.')[0] + '.jpg')
			elif(filename.split('.')[0] in val_images):
				writes_val.append('./images/val/' + filename.split('.')[0] + '.jpg')
	with open('data/VOC2012/val.txt', 'w') as f:
		for i in writes_val:
			f.write(i + '\n')
	with open('data/VOC2012/train.txt', 'w') as f:
		for i in writes_train:
			f.write(i + '\n')


		
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/bdd100k/labels/bdd100k_labels_images_val.json', help='*.json path')
    opt = parser.parse_args()
    conversion()