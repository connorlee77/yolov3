import sys
import os
import json
import argparse
from tqdm import tqdm

#1280x720
def conversion():
	names = []
	if not os.path.exists('out'):
		os.makedirs('out') 
	with open('data/coco.names') as f:
		for i in f:
			names.append(i[:-1])
	with open(opt.file) as f:
		data = json.load(f)
	writes = []
	for img in tqdm(data):
		img_name = img['name']
		img_name = str(img_name.split('.')[0]) + '.txt'
		with open('out/' + img_name, 'w') as f:
			for i in img['labels']:
				object_name = i['category']
				if(object_name == 'rider'):
					object_name = 'person'
				elif(object_name not in names):
					continue
				bbox = i['box2d']
				width = (bbox['x2']-bbox['x1'])/1280
				height = (bbox['y2']-bbox['y1'])/720
				center_x = bbox['x1']/1280 + width/2
				center_y = bbox['y1']/720 + height/2
				object_name = names.index(object_name)
				f.write(str(object_name) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height) + '\n')
			writes.append('./images/100k/val/' + img_name.split('.')[0] + '.jpg')
		with open('data/bdd100k/val.txt', 'w') as f:
			for i in writes:
				f.write(i + '\n')


		
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/bdd100k/labels/bdd100k_labels_images_val.json', help='*.json path')
    opt = parser.parse_args()
    conversion()