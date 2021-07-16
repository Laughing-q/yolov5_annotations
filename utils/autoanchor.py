# Auto-anchor utils

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import colorstr


def check_anchor_order(m):
    """
    检查anchor的顺序
    m：模型的Detect模块
    """
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    # 计算anchor的面积
    # anchor shape应该是(3, 3, 2), 如果是p6模型则是(4, 3, 2)
    # a (9, )/(12, )
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    # 计算最大与最小anchor面积差
    da = a[-1] - a[0]  # delta a
    # 计算最大与最小stride差
    ds = m.stride[-1] - m.stride[0]  # delta s
    # torch.sign(x):当x大于/小于0时，返回1/-1
    # 如果这里anchor与stride顺序不一致，则重新调整顺序
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """检查anchor是否满足要求
    dataset：标签
    model：模型
    thr:anchor与标签框的比例范围为(1/thr, thr), 在此范围内算是满足要求
    imgz:模型输入尺寸
    """
    # Check anchor fit to data, recompute if necessary
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    # reshape 图片的image_size为 基于imgsz为最大边的 大小
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 随机获取scale
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    # 将标签框的归一化坐标缩放为 基于图片大小为shapes * scale的坐标
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """根据wh计算anchor是否满足要求，即anchor与标签框的比值要在(1/thr, thr)
        k:一般传入的是anchor, [M, 2]
        wh: (N, 2)
        """
        # (N, 1, 2) / (1, M, 2) -> (N, M, 2)
        r = wh[:, None] / k[None]
        # (N, M), 找到w与h较小的比值, torch.min(2)按轴返回最小值和对应索引
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # (N, ), 找到每个wh与anchor的最大比值
        best = x.max(1)[0]  # best_x
        # 平均每个标签框有多少个满足要求的anchor与之对应
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        # 最大的比值大于1/thr满足要求即可，即代表该标签框有至少一个满足要求的anchor，bpr是datasets中满足要求的比例
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # current anchors
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    # 如果比例小于0.98则重新kmeans聚类生成新anchor
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        try:
            # 生成新的anchor
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        # 计算新anchor满足要求的比例, 如果大于之前的比例，则进行替换
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    # 注意一下下面的thr不是传入的thr，而是1/thr, 所以在计算指标这方面还是和check_anchor一样
    thr = 1. / thr
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        """输出新anchor的指标结果"""
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    # reshape 图片的image_size为 基于imgsz为最大边的 大小
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 将标签框的归一化坐标缩放为 基于图片大小为shapes的坐标
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    # 筛选长宽小于3的标签框
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    # 筛选长宽大于2的标签框
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    # kmeans 聚类出n个点，n就是anchor的数量
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    # 根据聚类出来的n个点采用遗传算法生成新的anchor
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)
