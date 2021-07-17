"""Test a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/test.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a cocoapi-compatible JSON results file
        project='runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        wandb_logger=None,
        compute_loss=None,
        ):
    """
    :param model: 测试的模型，训练时调用val传入
    :param dataloader: 测试集的dataloader，训练时调用val传入
    :param save_dir: 保存在测试时第一个batch的图片上画出标签框和预测框的图片路径
    :param plots: 是否绘制各种可视化，比如测试预测，混淆矩阵，PR曲线等
    :param wandb_looger: wandb可视化工具, train的时候传入
    :param compute_loss: 计算损失的对象实例, train的时候传入
    """
    # Initialize/load model and set device
    # 判断是否在训练时调用val，如果是则获取训练时的设备
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        # 选择设备
        device = select_device(device, batch_size=batch_size)

        # Directories
        # 获取保存日志路径
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 加载模型
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        # 检查输入图片分辨率是否能被32整除
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        # 加载数据配置信息
        with open(data) as f:
            data = yaml.safe_load(f)
        check_dataset(data)  # check

    # Half
    # 如果设备不是cpu且opt.half=True，则将模型由Float32转为Float16，提高前向传播的速度
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    # 判断是否为coco数据集
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 设置iou阈值，从0.5~0.95，每间隔0.05取一次
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # iou个数
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        # 创建一个全0数组测试一下前向传播是否正常运行
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 注意这里rect参数为True，yolov5的测试评估是基于矩形推理的, 且有个0.5的填充
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    # 初始化测试的图片数量
    seen = 0
    # 声明混淆矩阵对象实例
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取类别的名字
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    """
    获取coco数据集的类别索引
    这里要说明一下，coco数据集有80个类别(索引范围应该为0~79)，
    但是他的索引却属于0~90
    coco80_to_coco91_class()就是为了与上述索引对应起来，返回一个范围在0~90的索引数组
    """
    coco91class = coco80_to_coco91_class()
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化指标，时间
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件的字典，统计信息，ap, wandb显示图片
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        """
        time_synchronized()函数里面进行了torch.cuda.synchronize()，再返回的time.time()
        torch.cuda.synchronize()等待gpu上完成所有的工作
        总的来说就是这样测试时间会更准确 
        """
        t_ = time_synchronized()
        img = img.to(device, non_blocking=True)
        # 图片也由Float32->Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_synchronized()
        # 计算数据拷贝，类型转换以及除255的时间
        t0 += t - t_

        # Run model
        # 前向传播
        # out为预测结果, train_out训练结果
        out, train_out = model(img, augment=augment)  # inference and training outputs
        # 计算推理时间
        t1 += time_synchronized() - t

        # Compute loss
        # 如果是在训练时进行的val，则通过训练结果计算并返回测试集的box, obj, cls损失
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        # 将归一化标签框反算到基于原图大小，如果设置save-hybrid则传入nms函数
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        # 进行nms
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # t2为nms时间
        t2 += time_synchronized() - t

        # Statistics per image
        # 为每一张图片做统计, 写入预测信息到txt文件, 生成json文件字典, 统计tp等
        for si, pred in enumerate(out):
            # 获取第si张图片的标签信息, 包括class,x,y,w,h
            # targets[:, 0]为标签属于哪一张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            # 获取标签类别
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            # 统计测试图片数量
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # 反算坐标 基于input-size -> 基于原图大小
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            # 保存预测结果为txt文件
            if save_txt:
                # 获得对应图片的长和宽
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    # xyxy格式->xywh, 并对坐标进行归一化处理
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # 保存预测坐标，类别，置信度
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel plots
            # wandb logging
            # 添加wandb信息，图片等
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            # 保存coco格式的json文件字典
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # coco格式json文件大概包含信息如上
                # 获取图片id
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                # box转换为xywh格式
                box = xyxy2xywh(predn[:, :4])  # xywh
                """
                值得注意的是，之前所说的xyxy格式为左上角右下角的坐标
                xywh是中心点坐标和长和宽
                而coco的json格式中的框坐标格式为xywh,此处的xy为左上角坐标
                也就是coco的json格式的坐标格式为：左上角坐标+长宽
                所以下面一行代码就是将：中心点坐标->左上角
                """
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                """
                image_id：图片id, 即属于哪张图
                category_id: 类别, coco91class()从索引0~79映射到索引0~90
                bbox：框的坐标
                score：置信度
                """
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            # 初始化预测评定，niou为iou阈值的个数
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # detected用来存放已检测到的目标
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                # 获得xyxy格式的框
                tbox = xywh2xyxy(labels[:, 1:5])
                # 反算坐标 基于input-size -> 基于原图大小
                # 值得注意的是在进行nms之前，将归一化的标签框反算为基于input-size的标签框了
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    # 混淆矩阵的处理
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                # 对图片中的每个类单独处理
                for cls in torch.unique(tcls_tensor):
                    # 标签框该类别的索引
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    # 预测框该类别的索引
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # box_iou计算预测框与标签框的iou值，max(1)选出最大的ious值,i为对应的索引
                        """
                        pred shape[N, 4]
                        tbox shape[M, 4]
                        box_iou shape[N, M]
                        ious shape[N, 1]
                        i shape[N, 1], i里的值属于0~M
                        """
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            # 获得检测到的目标
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                # 添加d到detected
                                detected_set.add(d.item())
                                detected.append(d)
                                # iouv为以0.05为步长 0.5到0.95的序列
                                # 获得不同iou阈值下的true positive
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # 画出前三个batch的图片的ground truth和预测框并保存
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # 根据上面得到的tp等信息计算指标
        # 精准度TP/TP+FP，召回率TP/P，map，f1分数，类别
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt是一个列表，测试集每个类别有多少个标签框
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    # 打印指标结果
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 细节展示每一个类别的指标
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印前处理时间，前向传播耗费的时间、nms的时间
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        # 绘制混淆矩阵
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # wandb 显示信息，图片等
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('val*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    # 采用之前保存的json格式预测结果，通过cocoapi评估指标
    # 需要注意的是 测试集的标签也需要转成coco的json格式
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # 获取标签json文件路径
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 获取预测框的json文件路径并保存
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # 加载标签json文件, 预测json文件
            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    # 返回测试指标结果
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """
    opt参数详解
    data:数据集配置文件，数据集路径，类名等
    weights:测试的模型权重文件
    batch-size:前向传播时的批次, 默认32
    imgsz:输入图片分辨率大小, 默认640
    conf-thres:筛选框的时候的置信度阈值, 默认0.001
    iou-thres:进行NMS的时候的IOU阈值, 默认0.65
    task:设置测试形式, 默认val, 具体可看下面代码解析注释
    device:测试的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    single-cls:数据集是否只有一个类别，默认False
    verbose:是否打印出每个类别的mAP, 默认False
    save-txt:是否以txt文件的形式保存模型预测的框坐标, 默认False
    save-hybrid:是否将label与pred一起保存到txt文件中，默认False
    save-conf:是否将置信度conf也保存到txt中，默认False
    save-json:是否按照coco的json格式保存预测框，并且使用cocoapi做评估(需要同样coco的json格式的标签), 默认False
    project:保存测试日志的文件夹路径
    name:保存测试日志文件夹的名字, 所以最终是保存在project/name中
    exist_ok: 是否重新创建日志文件, False时重新创建文件
    half:是否使用F16精度推理
    """
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    # 设置参数save_json
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    # check_file检查文件是否存在
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    # 初始化logging
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # 检查环境
    check_requirements(exclude=('tensorboard', 'thop'))

    # task = 'val', 'test', 'study', 'speed'
    # task = ['val', 'test']时就正常测试验证集、测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    # 评估模型速度
    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)

    # task == 'study'时，就评估模型在各个尺度下的指标并可视化
    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
