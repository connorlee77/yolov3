from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import json

from torch.utils.data import DataLoader
from natsort import natsorted
from models import *
from utils.datasets import *
from utils.utils import *
import tqdm

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import matplotlib.pyplot as plt 


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


#########################################
### Replace with script to get imgIds ###
#########################################
imgsz=512
path = 'data/mammoth.txt'
dataset = LoadImagesAndLabels(path, imgsz, 1, rect=False, single_cls=False, pad=0.5)
dataloader = DataLoader(dataset)

imgIds = [int(Path(x).stem.split('_')[-1].replace('frame', '').replace('thumb', '')) for x in dataloader.dataset.img_files]
imgIds.sort()
#########################################
#########################################


# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
cocoGt = COCO(glob.glob('coco/annotations/instances_mammoth.json')[0])  # initialize COCO ground truth api
# cocoGt = COCO(glob.glob('coco/annotations/instances_wrightwood.json')[0])  # initialize COCO ground truth api
# cocoGt = COCO(glob.glob('coco/annotations/instances_val2017.json')[0])  # initialize COCO ground truth api


######################################
### Replace with SSD results array ###
######################################
cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api
######################################
######################################

results = np.zeros((len(imgIds), 4))
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
for i in tqdm.tqdm(range(len(imgIds))):
	cocoEval.params.imgIds = imgIds[i]  # only evaluate these images

	with suppress_stdout_stderr():
		cocoEval.evaluate()
		cocoEval.accumulate()
		cocoEval.summarize()

	stats = cocoEval.stats

	ap50_95, ap50, ap75, ap50_95_sm, ap50_95_md, ap50_95_lg = stats[0:6]
	# print(ap50)
	results[i, 0] = imgIds[i]
	results[i, 1] = ap50_95
	results[i, 2] = ap50
	results[i, 3] = ap75

np.save('mammoth_precisions', results)
plt.figure(figsize=(15,4))
plt.scatter(results[:,0], results[:,2], s=2)
plt.xlabel('Frame #')
plt.ylabel('Avg. Frame Precision')
plt.savefig('mammoth_precisions.svg')
plt.show()