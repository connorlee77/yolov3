import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2 

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(1024, 80)

#     # x represents our data
#     def forward(self, x):
#         print(x.shape)
#         x = self.fc1(x.squeeze())
#         return x

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # self.conv1 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, padding_mode='reflect')
      # self.GAP =  nn.AdaptiveAvgPool2d((1,1))
      self.fc1 = nn.Linear(1024, 512, bias=True)
      self.fc2 = nn.Linear(512, 80)

    # x represents our data
    def forward(self, x):
        B, C, H, W = x.shape
        # x = self.conv1(x)
        # x = self.GAP(x).view(B, -1)
        x = self.fc1(x.view(B,-1))
        x = F.relu(x)
        # x = x.view(B, -1)
        x = self.fc2(x)
        return x



def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample):
    # generate the class activation maps upsample to 256x256
    nc, h, w = feature_conv.shape
    output_cam = []
    unnormalized_cams = []

    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cv2.resize(cam, size_upsample[::-1], interpolation=cv2.INTER_CUBIC)
        temp_cam = cam[:]
        cam = np.maximum(cam, 0)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        
        output_cam.append(cam_img)
        unnormalized_cams.append(temp_cam)
    
    return output_cam, unnormalized_cams

def gradCAM(base_model, top_net, img, index, device, size_upsample):
    # generate the class activation maps upsample to 256x256
    data = base_model(img, augment=opt.augment)
    pred, image_path, features = data
    features.retain_grad()

    features_pooled =  nn.AdaptiveAvgPool2d((1,1)) (features)
    scores = top_net(features_pooled)

    base_model.zero_grad()
    top_net.zero_grad()

    # one_hot = torch.zeros(scores.size()).to(device)
    # one_hot[0,index] = 1
    # one_hot_scores = torch.sum(one_hot*scores)
    scores[0,index].backward()

    alpha = nn.AdaptiveAvgPool2d((1,1)) (features.grad)
    cam = torch.zeros(features.shape[2:]).to(device)

    for i, w in enumerate(alpha.squeeze()):
        cam += w*features[0,i]


    cam = F.relu(cam).cpu().detach().numpy()
    cam = cv2.resize(cam, size_upsample[::-1], interpolation=cv2.INTER_CUBIC)
    
    # cam = cam - np.min(cam)
    # cam = cam / np.max(cam)

    # cam = np.uint8(255*cam)

    return cam


def gradCAMplusplus(base_model, top_net, img, index, device, size_upsample):
    # generate the class activation maps upsample to 256x256
    data = base_model(img, augment=opt.augment)
    pred, image_path, features = data
    
    B, C, H, W = features.shape
    features.retain_grad()

    features_pooled =  nn.AdaptiveAvgPool2d((1,1)) (features)
    scores = top_net(features_pooled)

    base_model.zero_grad()
    top_net.zero_grad()

    scores[0,index].backward()

    A = features
    dSdA = A.grad
    dYdA = torch.exp(scores[0,index]) * dSdA

    numerator = dSdA.pow(2)
    denominator = 2*dSdA.pow(2) + A.sum(dim=(2,3)).view(1,C,1,1)*dSdA.pow(3)
    # denominator = torch.where(torch.abs(denominator) < 1e-6, denominator, torch.ones_like(denominator))
    alpha = numerator / (denominator + 1e-8) 

    weights = torch.sum(alpha * dYdA, dim=(2,3))

    cam = torch.sum(weights.view(1,C,1,1)*features, dim=1).squeeze()

    cam = F.relu(cam).cpu().detach().numpy()
    cam = cv2.resize(cam, size_upsample[::-1], interpolation=cv2.INTER_LINEAR)
    # cam = np.maximum(cam, 0)
    # cam = cam - np.min(cam)
    # cam = cam / np.max(cam)
    # cam = cam / 30

    # cam = np.uint8(255*cam)

    return cam



def make_folder(out):
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder


def detect(save_img=False):
    
    top_net = Net()
    top_net.load_state_dict(torch.load('focal_fcn_best_test_weights.pt'))

    params = list(top_net.named_parameters())
    # print(params[-2][1].shape)
    top_net_weights = params[-2][1].data.cpu().numpy()


    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    top_net.to(device)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()
    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        print(imgsz)
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    names.append('aux')

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
        index = 0
        t1 = torch_utils.time_synchronized()
        # img.requires_grad = True
        # data = model(img, augment=opt.augment)
        # pred, image_path, features = data
        # features =  nn.AdaptiveAvgPool2d((1,1)) (features)
        # predictions = top_net(features)
        # labels = torch.sigmoid(predictions) > 0.75
        # # labels = torch.ones().to(device)
        # criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # loss = criterion(predictions, labels.float())
        # model.zero_grad()
        # top_net.zero_grad()
        # loss.backward()

        # gradient =  torch.ge(img.grad, 0)
        # gradient = (gradient.float() - 0.5) * 2
        # temp_img = img - 0.0014*gradient
        ## Perturb gradient ##

        # data = model(temp_img, augment=opt.augment)
        # pred, image_path, temp_features = data
        # temp_features =  nn.AdaptiveAvgPool2d((1,1)) (temp_features)
        # temp_predictions = top_net(temp_features)

        data = model(img, augment=opt.augment)
        pred, image_path, features = data
        features =  nn.AdaptiveAvgPool2d((1,1)) (features)
        predictions = top_net(features)

        class_idx = list(range(0,80))
        size_upsample = im0s.shape[0:2]

        cam = gradCAM(model, top_net, img, index, device, size_upsample)
        # temp_cam = gradCAMplusplus(model, top_net, temp_img, index, device, size_upsample)
        # cam = gradCAM(model, top_net, img, 0, device, size_upsample)

        
        # cam, unnormalized_cams = returnCAM(features.detach().cpu().numpy().squeeze(), top_net_weights, class_idx, size_upsample)

        result = []
        predictions = torch.sigmoid(predictions).flatten()
        for i in range(80):
            if predictions[i] > 0.25:
                result.append([names[i], predictions[i].item()])
        print(result)

        # result = []
        # temp_predictions = torch.sigmoid(temp_predictions).flatten()
        # for i in range(80):
        #     if temp_predictions[i] > 0.25:
        #         result.append([names[i], temp_predictions[i].item()])
        # print(result)

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

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # print(xyxy)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))


            # Save results (image with detections)
            if save_img:

                
                # norm_factor = max([np.max(cam), np.max(temp_cam)])

                cam -= np.min(cam)
                norm_factor = np.max(cam)
                heatmap = cv2.applyColorMap(np.uint8(255*cam/norm_factor), cv2.COLORMAP_JET)

                new_img = heatmap*0.3 + im0*0.5
                new_folder = os.path.join(out, names[index])
                make_folder(new_folder)
                cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)

                # heatmap = cv2.applyColorMap(np.uint8(255*temp_cam/norm_factor), cv2.COLORMAP_JET)

                # new_img = heatmap*0.3 + im0*0.5
                # new_folder = os.path.join(out + '_odin', names[index])
                # make_folder(new_folder)
                # cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)

                # subtract_cam = np.abs(cam - temp_cam)
                # print(np.max(subtract_cam))
                # subtract_cam /= np.max(subtract_cam)
                # # subtract_cam /= 20
                # subtract_cam = np.uint8(255*subtract_cam)

                # heatmap = cv2.applyColorMap(subtract_cam, cv2.COLORMAP_JET)

                # new_img = heatmap*0.3 + im0*0.5
                # new_folder = os.path.join(out + '_subtract', names[index])
                # make_folder(new_folder)
                # cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)

                    # unnormalized_heatmap = unnormalized_cams[i]

                    # print(unnormalized_heatmap.shape)
                    # np.savetxt(os.path.join(new_folder, os.path.basename(save_path).replace('png', 'txt').replace('jpg', 'txt')), unnormalized_heatmap)


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
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)
    detect()
    # with torch.no_grad():
        

