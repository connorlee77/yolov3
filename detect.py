import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2 

import kornia.feature
from bbox import find_paws, slice_to_bbox, remove_overlaps
import matplotlib.patches as patches
from scipy.io import savemat


class Net(nn.Module):
    def __init__(self, base_model):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, padding_mode='reflect')
        # self.GAP =  nn.AdaptiveAvgPool2d((1,1))

        self.activations = {}
        def get_activation(layer):
            def hook(module, input, output):
                self.activations[layer] = output
            return hook

        self.base_model = base_model

        for target_layer in [80, 92, 104]:
            layer = self.base_model.module_list[target_layer]
            layer.register_forward_hook(get_activation(target_layer))

        self.GAP =  nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(1024 + 512 + 256, 1024)
        self.fc2 = nn.Linear(1024, 80)

    # x represents our data
    def forward(self, x):
        B, C, H, W = x.shape
        
        sig_pred, pred, _ = self.base_model(x)

        act_80 = self.activations[80]
        act_92 = self.activations[92]
        act_104 = self.activations[104]
        
        act_80 = F.upsample(act_80, size=act_104.shape[2:4], mode='bilinear', align_corners=False)
        act_92 = F.upsample(act_92, size=act_104.shape[2:4], mode='bilinear', align_corners=False)

        x1 = torch.cat([act_80, act_92, act_104], dim=1)
        x = self.GAP(x1)

        x = self.fc1(x.view(B,-1))
        x = F.relu(x)
        x = self.fc2(x)
        return sig_pred, x, x1


def gradCAM(base_model, top_net, img, index, device, size_upsample):
    # generate the class activation maps upsample to 256x256
    data = base_model(img)
    pred, x, features = data
    features.retain_grad()
    base_model.zero_grad()

    x[0,index].backward()
    alpha = nn.AdaptiveAvgPool2d((1,1)) (features.grad)
    
    B, C, H, W = features.shape
    cam = alpha.view(1, C, 1, 1) * features
    cam = torch.sum(cam, dim=1).view(1, 1, H, W)

    cam = F.relu(cam)
    cam = F.interpolate(cam, size_upsample, mode='bilinear')
    
    # cam = cam - np.min(cam)
    # cam = cam / np.max(cam)
    # cam = np.uint8(255*cam)

    return cam


def make_folder(out):
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder


def detect(save_img=False):

    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    model = Net(base_model=model)
    model.load_state_dict(torch.load('SGD_B001_test_weights.pt'))

    # Load weights
    # attempt_download(weights)
    # if weights.endswith('.pt'):  # pytorch format
    #     model.load_state_dict(torch.load(weights, map_location=device)['model'])
    # else:  # darknet format
    #     load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    save_img = True
    print(imgsz)
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        fname = os.path.basename(path).split('.')[0]

        cv2.imwrite('resized_input_{}/{}.png'.format(out, fname), img.transpose(1, 2, 0))

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        index = opt.coco_index
        t1 = torch_utils.time_synchronized()
        img.requires_grad = True
        data = model(img)
        pred, x, features = data
        labels = torch.sigmoid(x) >= 0
        # labels = torch.ones().to(device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterion(x[0, :], labels[0, :].float())
        model.zero_grad()
        loss.backward()

        # gradient = img.grad
        gradient = torch.ge(img.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
        temp_img = img - 0.0014*gradient
        ## Perturb gradient ##

        pred, temp_predictions, temp_features = model(temp_img)

        data = model(img)
        pred, x, features = data

        # class_idx = list(range(0,80))
        size_upsample = im0s.shape[0:2]

        cam = gradCAM(model, None, img, index, device, size_upsample)
        temp_cam = gradCAM(model, None, temp_img, index, device, size_upsample)

        # result = []
        # predictions = torch.sigmoid(x).flatten()
        # for i in range(80):
        #     if predictions[i] > 0.5:
        #         result.append([names[i], predictions[i].item()])
        # print(result)
        # # print(predictions[index])

        # result = []
        # temp_predictions = torch.sigmoid(temp_predictions).flatten()
        # for i in range(80):
        #     if temp_predictions[i] > 0.0:
        #         result.append([names[i], temp_predictions[i].item()])
        # print(result)
        # print(temp_predictions[index])
        # print(predictions[index] / temp_predictions[index])
        # predictions = torch.sigmoid(predictions).flatten()
        # temp_predictions = torch.sigmoid(temp_predictions).flatten()
        # result = []
        # for i in range(80):
        #     if predictions[i] > 0.75:
        #         result.append([names[i], abs(predictions[i].item() - temp_predictions[i].item())])
        # print(result)

        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            # if det is not None and len(det):
            #     # Rescale boxes from imgsz to im0 size
            #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            #     # Print results
            #     for c in det[:, -1].detach().unique():
            #         n = (det[:, -1] == c).sum()  # detections per class
            #         s += '%g %ss, ' % (n, names[int(c)])  # add to string

            #     # Write results
            #     for *xyxy, conf, cls in reversed(det):
            #         if save_txt:  # Write to file
            #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #             with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
            #                 file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

            #         if save_img or view_img:  # Add bbox to image
            #             label = '%s %.2f' % (names[int(cls)], conf)
            #             # print(xyxy)
            #             plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))


            # Save results (image with detections)
            if save_img:

                # denom_cam = cam.clone()
                # denom_cam[denom_cam < 1e-3] = 1e10
                # percent_increase = (temp_cam - cam) / denom_cam

                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # plt.imshow(percent_increase.squeeze().cpu().detach().numpy())
                # plt.colorbar()
                # plt.show()
                
                
                # percent_increase = 255*(percent_increase.squeeze().cpu().detach().numpy() > 1e-2)
                # data_slices = find_paws(percent_increase, smooth_radius = 20, threshold = 22)
                # bboxes = slice_to_bbox(data_slices)
                # for bbox in bboxes:
                #     xwidth = bbox.x2 - bbox.x1
                #     ywidth = bbox.y2 - bbox.y1
                #     p = patches.Rectangle((bbox.x1, bbox.y1), xwidth, ywidth,
                #                           fc = 'none', ec = 'red')
                #     ax.add_patch(p)

                # plt.show()


                # diff_cam = torch.abs(cam - temp_cam).squeeze().cpu().detach().numpy()
                # # plt.imshow(diff_cam)
                # # plt.colorbar()
                # # plt.show()
                # print(np.min(diff_cam), np.max(diff_cam))
                # # diff_cam -= np.min(diff_cam)
                # # diff_cam /= np.max(diff_cam)
                # diff_cam /= 0.01

                # heatmap = cv2.applyColorMap(np.uint8(255*diff_cam), cv2.COLORMAP_JET)

                # new_img = heatmap*0.3 + im0*0.5
                # new_folder = os.path.join(out + '_diff', names[index])
                # make_folder(new_folder)
                # cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)


                # cam = cam.squeeze().cpu().detach().numpy()
                # cam -= np.min(cam)
                # cam /= np.max(cam)

                # heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

                # new_img = heatmap*0.3 + im0*0.5
                # new_folder = os.path.join(out + '_cam', names[index])
                # make_folder(new_folder)
                # cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)

                # temp_cam = temp_cam.squeeze().cpu().detach().numpy()
                # temp_cam -= np.min(temp_cam)
                # temp_cam /= np.max(temp_cam)

                # heatmap = cv2.applyColorMap(np.uint8(255*temp_cam), cv2.COLORMAP_JET)

                # new_img = heatmap*0.3 + im0*0.5
                # new_folder = os.path.join(out + '_temp_cam', names[index])
                # make_folder(new_folder)
                # cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)



                save_file = os.path.basename(save_path).split('.')[0] + '.mat'

                diff_cam = torch.abs(cam - temp_cam).squeeze().cpu().detach().numpy()
                new_folder = os.path.join(out + '_diff', names[index])
                make_folder(new_folder)
                savemat(os.path.join(new_folder, save_file), mdict={'cam': diff_cam})


                cam = cam.squeeze().cpu().detach().numpy()
                new_folder = os.path.join(out + '_cam', names[index])
                make_folder(new_folder)
                savemat(os.path.join(new_folder, save_file), mdict={'cam': cam})

                # gradient_cam = temp_cam.squeeze().cpu().detach().numpy()
                # new_folder = os.path.join(out + '_gradient_cam', names[index])
                # make_folder(new_folder)
                # savemat(os.path.join(new_folder, save_file), mdict={'cam': gradient_cam})


    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--coco_index', type=int, default=0, help='inference size (pixels)')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)
    detect()
    # with torch.no_grad():
        

