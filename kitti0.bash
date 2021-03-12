# python detect.py --source=/home/fremont/ford/kitti/training/yolo/images_crops --device=0 --output=kitti_crops_out --coco_index=0
# python detect.py --source=/home/fremont/ford/kitti/training/yolo/images_crops --device=0 --output=kitti_crops_out --coco_index=2

python detect.py --source=/home/fremont/ford/kitti/training/yolo/images --device=0 --output=kitti_out --coco_index=2
python detect.py --source=/home/fremont/ford/kitti/training/yolo/4000 --device=0 --output=kitti_4k_out --coco_index=2