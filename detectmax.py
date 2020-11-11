import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2 
import torch

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(1024, 1, bias=True)

#     # x represents our data
#     def forward(self, x):
#         print(x.shape)
#         x = self.fc1(x.squeeze())
#         return x

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, padding_mode='reflect')
      self.GAP =  nn.AdaptiveAvgPool2d((1,1))
      self.fc1 = nn.Linear(2048, 80)

    # x represents our data
    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.conv1(x)
        x = self.GAP(x1).view(B, -1)
        x = self.fc1(x)
        return x



def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample):
    # generate the class activation maps upsample to 256x256
    nc, h, w = feature_conv.shape
    output_cam = []
    unnormalized_cams = []

    smallest = []
    biggest = []
    for idx in class_idx:

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        
        # cam = cv2.resize(cam, size_upsample[::-1], interpolation=cv2.INTER_CUBIC)
        temp_cam = cam[:]

        biggest.append(np.max(cam))
        smallest.append(np.min(cam))


        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        cam_img = np.uint8(255 * cam_img)
        
        output_cam.append(cam_img)
        unnormalized_cams.append(temp_cam)
    
    max_cams = np.stack(unnormalized_cams, axis=0)
    argmax_maxcam = np.argmax(max_cams, axis=0)
    max_cams = np.max(max_cams, axis=0)

    unnorm_maxcam = max_cams[:]
    max_cams = cv2.resize(max_cams, size_upsample[::-1], interpolation=cv2.INTER_CUBIC)

    # max_cams = max_cams - (-87)
    # max_cams = max_cams / (69 - (-87))
    max_cams = max_cams - np.min(max_cams)
    max_cams = max_cams / np.max(max_cams)
    max_cams = np.uint8(255 * max_cams)

    return output_cam, unnormalized_cams, (smallest, biggest), max_cams, unnorm_maxcam, argmax_maxcam


def make_folder(out):
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder


def detect(save_img=False):
    
    top_net = Net()
    top_net.load_state_dict(torch.load('surfboard_fcn_best_test_weights.pt'))

    params = list(top_net.named_parameters())
    # print(params)
    # print(params[-1][1].shape)
    top_net_weights = params[-2][1].data.cpu().numpy()


    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    top_net.to(device)

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    resized_input_out = 'resized_input_{}'.format(out)
    if os.path.exists(resized_input_out):
        shutil.rmtree(resized_input_out)  # delete output folder
    os.makedirs(resized_input_out)  # make new output folder

    features_out = 'yolov3_{}'.format(out)
    if os.path.exists(features_out):
        shutil.rmtree(features_out)  # delete output folder
    os.makedirs(features_out)  # make new output folder

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
    # names.append('aux')

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    smallest = np.inf
    biggest = 0
    for path, img, im0s, vid_cap in dataset:
        fname = os.path.basename(path).split('.')[0]

        cv2.imwrite('resized_input_{}/{}.png'.format(out, fname), img.transpose(1, 2, 0))

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        data = model(img, augment=opt.augment)

        pred, image_path, features = data

        features1 = top_net.conv1(features)
        
        class_idx = list(range(0,1))
        size_upsample = im0s.shape[0:2]
        cam, unnormalized_cams, (small, big), max_cam, unnorm_maxcam, argmax_maxcam = returnCAM(features1.cpu().numpy().squeeze(), top_net_weights, class_idx, size_upsample)
        smallest = min(smallest, min(small))
        biggest = max(biggest, max(big))

        # features =  nn.AdaptiveAvgPool2d((1,1)) (features)
        predictions = top_net(features)
        fname = os.path.basename(path).split('.')[0]
        np.save(os.path.join(features_out, '{}.npy'.format(fname)), features.cpu().numpy())

        result = []
        predictions = torch.sigmoid(predictions).flatten()
        for i in range(80):
            if predictions[i] > 0.7:
                result.append([names[i], predictions[i].item()])

        print(result)

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
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
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

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    w,h = size_upsample
                    heatmap = cv2.applyColorMap(max_cam, cv2.COLORMAP_JET)

                    new_img = heatmap*0.3 + im0*0.5

                    new_folder = os.path.join(out, 'max_cam')
                    make_folder(new_folder)
                    cv2.imwrite(os.path.join(new_folder, os.path.basename(save_path)), new_img)
                    np.savetxt(os.path.join(new_folder, os.path.basename(save_path).replace('png', 'txt').replace('jpg', 'txt')), unnorm_maxcam)
                    np.savetxt(os.path.join(new_folder, 'argmax_' + os.path.basename(save_path).replace('png', 'txt').replace('jpg', 'txt')), argmax_maxcam)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print(smallest, biggest)
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

    with torch.no_grad():
        detect()

