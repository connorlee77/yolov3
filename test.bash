python test.py --data=data/kitti.data --conf-thres=0.3 --iou-thres=0.4 --attenuation='' --device=0
python test.py --data=data/kitti4k.data --conf-thres=0.3 --iou-thres=0.4 --attenuation='4k_' --device=0
python test.py --data=data/kitti4k.data --conf-thres=0.3 --iou-thres=0.4 --attenuation='6k_' --device=0
python test.py --data=data/kitti7k.data --conf-thres=0.3 --iou-thres=0.4 --attenuation='7k_' --device=0
python test.py --data=data/kitti10k.data --conf-thres=0.3 --iou-thres=0.4 --attenuation='10k_' --device=0

python test.py --data=data/kittiOOD.data --conf-thres=0.001 --iou-thres=0.4 --attenuation='ood_' --device=0