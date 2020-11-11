import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

path = 'yolov3_mammoth_cluster_features'
files = glob.glob(os.path.join(path, '*.npy'))
label_path = '/home/fremont/ford/JSON2YOLO/out/labels/mammoth'

arr = []
label_arr = []
for f in files:
	array = np.load(f)
	arr.append(array.squeeze())
	
	label_file = os.path.basename(f).replace('npy', 'txt')

	label_filepath = os.path.join(label_path, label_file)
	label = np.zeros(80)
	if os.path.exists(label_filepath):
		label_txt = np.genfromtxt(label_filepath, delimiter=' ')
		if len(label_txt.shape) > 1:
			unique_labels = set(label_txt[:,0])
		else:
			unique_labels = [label_txt[0]]
		for i in unique_labels:
			label[int(i)] = 1

	label_arr.append(label)

data = np.stack(arr)
labels = np.stack(label_arr)

pca = PCA(n_components=50)
pca_result = pca.fit_transform(data)
print('pca result')
tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=5)
result = tsne.fit_transform(pca_result)
# result = pca_result
print(result.shape)
for i in tqdm.tqdm(range(len(result))):
	color = 'black'
	if labels[i,0] == 1 and labels[i,2] == 1 and labels[i,7] == 1:
		color = 'yellow'
	elif labels[i,0] == 1 and labels[i,2] == 1:
		color = 'orange'
	elif labels[i,2] == 1 and labels[i,7] == 1:
		color = 'cyan'
	elif labels[i,0] == 1 and labels[i,7] == 1:
		color = 'magenta'
	elif labels[i,0] == 1:
		color = 'red'
	elif labels[i,2] == 1:
		color = 'blue'
	elif labels[i,7] == 1:
		color = 'green'

	plt.scatter(result[i,0], result[i,1], c=color, s=4)

plt.show()