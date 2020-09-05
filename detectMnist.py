import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import torch
import torchvision
import tqdm
import PIL.ImageOps

def detect(save_img=False):
    
    imgsz = opt.img_size
    out, weights = opt.output, opt.weights
 
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    feature_out = 'yolov3_{}'.format(out)
    if os.path.exists(feature_out):
        shutil.rmtree(feature_out)  # delete output folder
    os.makedirs(feature_out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(imgsz), torchvision.transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                   download=True, transform=transform)

    fashion_mnist = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=False,
                                   download=True, transform=transform)

    dataset = fashion_mnist if opt.use_fashion else mnist

    target_count = {}
    # Run inference
    for i, (img, target) in enumerate(tqdm.tqdm(dataset)):
 
        if target not in target_count:
            target_count[target] = 1
        else:
            target_count[target] += 1
        temp = torchvision.transforms.ToPILImage()(img[0])
        # temp = PIL.ImageOps.invert(temp)
        temp = torchvision.transforms.ToTensor()(temp)
        img[0] = temp
        img = img.repeat(3, 1, 1).unsqueeze(0)
        img = img.to(device)

        # Inference
        data = model(img)

        pred, image_path, features = data

        if opt.global_avg_pool:
            features =  nn.AdaptiveAvgPool2d((1,1)) (features)
        if i == 1:
            print(features.shape)

        fname = '{}{}'.format(str(target), str(target_count[target]).zfill(5))
        np.save(os.path.join(feature_out, '{}.npy'.format(fname)), features.cpu().numpy())

        # Save results (image with detections)
        cv2.imwrite(os.path.join(out, '{}.png'.format(fname)), img.cpu().numpy().squeeze().transpose(1, 2, 0)*255)


    print(target_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='weights path')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--global_avg_pool', action='store_true')
    parser.add_argument('--use_fashion', action='store_true')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    print(opt)

    with torch.no_grad():
        detect()

