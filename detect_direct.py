import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2 



class GradCam(nn.Module):

    def __init__(self, model, target_layers):
        super(GradCam, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.base_model = model 
        self.target_layers = target_layers

        self.gradients = dict()
        self.activations = dict()

        def get_gradient(layer):
            def hook(module, grad_input, grad_output):
                self.gradients[layer] = grad_output[0]
            return hook

        def get_activation(layer):
            def hook(module, input, output):
                self.activations[layer] = output
            return hook

        for target_layer in self.target_layers:
            layer = self.base_model.module_list[target_layer]
            layer.register_backward_hook(get_gradient(target_layer))
            layer.register_forward_hook(get_activation(target_layer))

    def forward(self, img, index, device, size_upsample):
        # generate the class activation maps upsample to 256x256
        pred, p, features = self.base_model(img, augment=opt.augment)        
        # filtered_pred_agg = torch.sum(pred, dim=1).squeeze()

        self.base_model.zero_grad()

        pred_nosigmoid = [px.view(1, -1, 85) for px in p]
        pred_nosigmoid = torch.cat(pred_nosigmoid, dim=1)
        pred_nosigmoid = pred_nosigmoid[0, pred_nosigmoid[0,:,4] >= 0]
        pred_nosigmoid = pred_nosigmoid.sum(dim=0)
        
        pred_nosigmoid[5+index].backward()

        total_cam = torch.zeros(size_upsample).to(device)
        cam_max = []
        for layer_n in self.target_layers:

            alpha_n = self.gap(self.gradients[layer_n])
            features_n = self.activations[layer_n]

            B, C, H, W = features_n.shape
            cam = alpha_n.view(1, C, 1, 1) * features_n
            cam = torch.sum(cam, dim=1).view(1, 1, H, W)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size_upsample, mode='bilinear')
            # cam = cam / (torch.max(cam) + 1e-8)

            # cam[cam > np.quantile(cam.cpu().detach().numpy(), 0.99)] = 1
            
            total_cam += cam.squeeze()

            cam_max.append(cam)

        cam_max, _ = torch.cat(cam_max, dim=0).squeeze().max(dim=0)   
        cam_max = cam_max.cpu().detach().numpy()

        return cam_max
        total_cam /= len(self.target_layers)          
        total_cam = total_cam.cpu().detach().numpy()
        return total_cam

    # def forward1(self, img, index, device, size_upsample):
    #     pred, p, features = self.base_model(img, augment=opt.augment)        
    #     # filtered_pred_agg = torch.sum(pred, dim=1).squeeze()

    #     self.base_model.zero_grad()
    #     pred_nosigmoid = [px.view(1, -1, 85) for px in p]
    #     pred_nosigmoid = torch.cat(pred_nosigmoid, dim=1)
    #     pred_nosigmoid = pred_nosigmoid[0, pred_nosigmoid[0,:,4] >= 0]
    #     pred_nosigmoid = pred_nosigmoid.mean(dim=0)
        
    #     pred_nosigmoid[5+index].backward()

    #     cams = []
    #     total_cam = torch.zeros(size_upsample).to(device)
    #     for layer_n in self.target_layers:
    #         A = self.activations[layer_n]
    #         B, C, H, W = A.shape

    #         dsda = self.gradients[layer_n]
    #         dyda = torch.exp(pred_nosigmoid[5+index]) * dsda 
    #         numerator = dsda.pow(2)
    #         denominator = 2*dsda.pow(2) + A.sum(dim=(2,3)).view(1,C,1,1)*dsda.pow(3) + 1e-8
    #         # denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))

    #         alpha = numerator / denominator
    #         alpha_norm_constant = alpha.sum()
    #         alpha /= alpha_norm_constant


    #         weights = torch.sum(alpha * F.relu(dyda), dim=(2,3))
    #         cam = torch.sum(weights.view(1,C,1,1)*A, dim=1).view(1, 1, H, W)
    #         cam = F.relu(cam)
    #         cam = F.interpolate(cam, size_upsample, mode='bilinear')
    #         # cams.append(cam)

    #         total_cam += cam.squeeze()

    #     total_cam = total_cam / len(self.target_layers)
    #     # cams = torch.cat(cams, dim=0).squeeze()
    #     # total_cam, _ = torch.max(cams, dim=0)

    #     return total_cam.cpu().detach().numpy()





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

    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

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

    gradcam = GradCam(model, target_layers=[80, 92, 104])

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True
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
        data = model(img, augment=opt.augment)
        pred, image_path, features = data
        labels = pred[0,:,index] > 0.75
        # labels = torch.ones().to(device)
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(pred[0,:,index], labels.float())
        model.zero_grad()
        loss.backward()

        gradient =  torch.ge(img.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
        temp_img = img - 0.0014*gradient


        data = model(img, augment=opt.augment)
        pred, image_path, features = data
        size_upsample = im0s.shape[0:2]

        cam = gradcam(img, index, device, size_upsample)

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
    parser.add_argument('--coco_index', type=int, default=0, help='inference size (pixels)')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)
    detect()
    # with torch.no_grad():
        

