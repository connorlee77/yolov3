import argparse
import json

from torch.utils.data import DataLoader
from natsort import natsorted
from models import *
from utils.datasets import *
from utils.utils import *

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import matplotlib.pyplot as plt 

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):
    # Initialize/load model and set device
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5, augment=True, hyp=hyp)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for idx in range(10):
        for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = imgs.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)
            
            # Disable gradients
            with torch.no_grad():
                # Run model
                t = torch_utils.time_synchronized()
                inf_out, train_out, features = model(imgs, augment=False)  # inference and training outputs

                features =  nn.AdaptiveAvgPool2d((1,1)) (features)
                BATCH_N = features.shape[0]
                for i in range(BATCH_N):
                    path = paths[i]
                    fname = os.path.basename(path).split('.')[0]
                    print(features[i].cpu().numpy().shape)
                    npy_path = os.path.join(out, '{}_{}.npy'.format(fname, str(idx).zfill(2)))
                    np.save(npy_path, features[i].cpu().numpy())

                # t0 += torch_utils.time_synchronized() - t

                # # Compute loss
                # if is_training:  # if model has loss hyperparameters
                #     loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

                # # Run NMS
                # t = torch_utils.time_synchronized()
                # output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
                # t1 += torch_utils.time_synchronized() - t

            # Statistics per image
            # for si, pred in enumerate(output):
            #     labels = targets[targets[:, 0] == si, 1:]
            #     nl = len(labels)
            #     tcls = labels[:, 0].tolist() if nl else []  # target class
            #     seen += 1

            #     if pred is None:
            #         if nl:
            #             stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            #         continue

            #     # Append to text file
            #     # with open('test.txt', 'a') as file:
            #     #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            #     # Clip boxes to image bounds
            #     clip_coords(pred, (height, width))

            #     # Append to pycocotools JSON dictionary
            #     if save_json:
            #         # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            #         image_id = int(Path(paths[si]).stem.split('_')[-1].replace('frame', '').replace('thumb', ''))
            #         box = pred[:, :4].clone()  # xyxy
            #         scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
            #         box = xyxy2xywh(box)  # xywh
            #         box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            #         for p, b in zip(pred.tolist(), box.tolist()):
            #             jdict.append({'image_id': image_id,
            #                           'category_id': coco91class[int(p[5])],
            #                           'bbox': [round(x, 3) for x in b],
            #                           'score': round(p[4], 5)})

            #     # Assign all predictions as incorrect
            #     correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            #     if nl:
            #         detected = []  # target indices
            #         tcls_tensor = labels[:, 0]

            #         # target boxes
            #         tbox = xywh2xyxy(labels[:, 1:5]) * whwh

            #         # Per target class
            #         for cls in torch.unique(tcls_tensor):
            #             ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
            #             pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

            #             # Search for detections
            #             if pi.shape[0]:
            #                 # Prediction to target ious
            #                 ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

            #                 # Append detections
            #                 for j in (ious > iouv[0]).nonzero():
            #                     d = ti[i[j]]  # detected target
            #                     if d not in detected:
            #                         detected.append(d)
            #                         correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
            #                         if len(detected) == nl:  # all targets already located in image
            #                             break



                # Append statistics (correct, conf, pcls, tcls)
                # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                
                # arr = [(correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)]
                # arr = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
                # if len(arr):
                #     p1, r1, ap1, f11, ap_class1 = ap_per_class(*arr)
                #     if niou > 1:
                #         p1, r1, ap1, f11 = p1[:, 0], r1[:, 0], ap1.mean(1), ap1[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
                #     mp1, mr1, map1, mf11 = p1.mean(), r1.mean(), ap1.mean(), f11.mean()
                #     # print(map1)

            # Plot images
    #         if batch_i == 1:
    #             f = 'test_batch%g_gt.jpg' % batch_i  # filename
    #             plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
    #             f = 'test_batch%g_pred.jpg' % batch_i
    #             plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)  # predictions

    # # Compute statistics
    # stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # if len(stats):
    #     p, r, ap, f1, ap_class = ap_per_class(*stats)
    #     if niou > 1:
    #         p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
    #     mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    #     nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    # else:
    #     nt = torch.zeros(1)

    # # Print results
    # pf = '%20s' + '%10.3g' * 6  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # # Print results per class
    # if verbose and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # # Print speeds
    # if verbose or save_json:
    #     t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    #     print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


    # # Save JSON
    # if save_json and map and len(jdict):
    #     print('\nCOCO mAP with pycocotools...')
    #     imgIds = [int(Path(x).stem.split('_')[-1].replace('frame', '').replace('thumb', '')) for x in dataloader.dataset.img_files]
    #     imgIds.sort()

    #     with open('{}.json'.format(opt.dataset_name), 'w') as file:
    #         json.dump(jdict, file)

    #     from pycocotools.coco import COCO
    #     from pycocotools.cocoeval import COCOeval

    #     # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #     cocoGt = COCO(glob.glob('coco/annotations/instances_{}.json'.format(opt.dataset_name))[0])
    #     cocoDt = cocoGt.loadRes('{}.json'.format(opt.dataset_name))  # initialize COCO pred api

    #     results = np.zeros((len(imgIds), 4))
    #     cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #     cocoEval.params.imgIds = imgIds  # only evaluate these images

    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()



        # for i in tqdm(range(len(imgIds))):
        #     cocoEval.params.imgIds = imgIds[i]  # only evaluate these images

        #     with suppress_stdout_stderr():
        #         cocoEval.evaluate()
        #         cocoEval.accumulate()
        #         cocoEval.summarize()

        #     stats = cocoEval.stats

        #     ap50_95, ap50, ap75, ap50_95_sm, ap50_95_md, ap50_95_lg = stats[0:6]
        #     # print(ap50)
        #     results[i, 0] = imgIds[i]
        #     results[i, 1] = ap50_95
        #     results[i, 2] = ap50
        #     results[i, 3] = ap75

        # np.save('{}_precisions'.format(opt.dataset_name), results)
        # plt.figure(figsize=(15,4))
        # plt.scatter(results[:,0], results[:,2], s=2)
        # # plt.plot(results[:,0], results[:,2], '--')
        # plt.xlabel('Frame #')
        # plt.ylabel('Avg. Frame Precision')
        # plt.savefig('{}_ap50.svg'.format(opt.dataset_name))


    # Return results
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    # return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--dataset_name', type=str, default='test', help='json file and folder names')  # output folder
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)

    out = opt.output
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment)

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')
