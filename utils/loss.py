# Loss functions

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """标签平滑"""
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Generalized Focal loss中的Quality Focal Loss"""
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """计算损失"""
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        # 在计算objectness的时候是否对ciou进行排序
        self.sort_obj_iou = False
        # 获取设备
        device = next(model.parameters()).device  # get model device
        # 获取超参数
        h = model.hyp  # hyperparameters

        # Define criteria
        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 标签平滑，eps默认为0，其实是没用上。
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        # 如果设置了fl_gamma参数，就使用focal loss，默认也是没使用的
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # 获取模型的Detect层
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        # 设置三个特征图对应输出的损失系数
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        # 复制det的属性
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        # 初始化各个部分损失
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 获得标签分类，边框，索引，anchor
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # 根据indices获取索引，方便找到对应网格的输出
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # 找到对应网格的输出
                """
                这一步只取与标签所在网格的预测值来回归，
                从这里可以看出yolov5是根据网格来分配标签与预测(没有gt与预测框的iou计算)；
                """
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                """
                对输出xywh做反算
                值得注意的是，这里的tbox是基于FPN特征图输出网格大小的，
                比如说如果input-size为640，这里tbox是基于20x20、40x40、80x80的
                由于anchors在yolo.py/Model中是除以了stride，
                所以这里的预测pbox也是反算到基于FPN特征图输出网格大小的
                """
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 计算边框损失，注意这个CIoU=True，计算的是ciou损失
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                # 根据pbox与tbox的ciou对预测进行排序
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # 根据model.gr设置objectness的标签值
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                # 设置如果类别数大于1才计算分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # 计算objectness的损失
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            # 如果设定autobalance，则根据设定规则自动设置balance
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        # 如果设定autobalance，则根据设定规则自动设置balance
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 根据超参数设置的各个部分损失的系数 获取最终损失
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p: 网络输出，List[torch.tensor * 3], p[i].shape = (b, 3, h, w, nc+5), hw分别为特征图的长宽,b为batch-size
            targets: targets.shape = (nt, 6) , 6=icxywh,i表示第一张图片，c为类别，然后为坐标xywh
            model: 模型

        Returns:

        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # anchor数量和标签框数量
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ai.shape = (na, nt) 生成anchor索引
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets.shape = (na, nt, 7)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # 设置偏移量
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 对每个检测层进行处理
        for i in range(self.nl):
            anchors = self.anchors[i]
            # 得到特征图的坐标系数
            """
            p[i].shape = (b, 3, h, w，nc+5), hw分别为特征图的长宽
            gain = [1, 1, w, h, w, h, 1]
            """
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                """
                预测的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的(这应该是yolov5的创新点)，从而更好的回归(与新的边框回归方式有关)
                由于yolov3回归wh采用的是out=exp(in)，这很危险，因为out=exp(in)可能会无穷大，就会导致失控的梯度，不稳定，NaN损失并最终完全失去训练；
                (当然原yolov3采用的是将targets进行反算来求in与网络输出的结果，就问题不大，但采用iou loss，就需要将网络输出算成out来进行loss求解，所以会面临这个问题)；
                所以作者采用新的wh回归方式:
                (wh.sigmoid() * 2) ** 2 * anchors[i], 原来yolov3为anchors[i] * exp(wh)
                将标签框与anchor的倍数控制在0~4之间；
                hyp.scratch.yaml中的超参数anchor_t=4，所以也是通过此参数来判定anchors与标签框契合度；
                """
                # 计算比值ratio
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                """
                筛选满足1 / hyp['anchor_t'] < targets_wh/anchor_wh < hyp['anchor_t']的框;
                由于wh回归公式中将标签框与anchor的倍数控制在0~4之间，所以这样筛选之后也会浪费一些输出空间；
                由于分给每个特征金字塔层的anchor尺度都不一样，这里根据标签wh与anchor的wh的比例分配标签，
                就相当于把不同尺度的GT分配给了不同的特征层来回归；
                """
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # yolov5不再通过iou来分配标签，而仅仅使用网格分配；
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # 筛选过后的t.shape = (M, 7),M为筛选过后的数量
                t = t[j]  # filter

                # Offsets
                # 得到中心点坐标xy(相对于左上角的), (M, 2)
                gxy = t[:, 2:4]  # grid xy
                # 得到中心点相对于右下角的坐标, (M, 2)
                gxi = gain[[2, 3]] - gxy  # inverse
                # ((gxy % 1. < g) & (gxy > 1.)).T shape为(2, M)
                # jklm shape为(M, )
                """
                把相对于各个网格左上角x<0.5,y<0.5和相对于右下角的x<0.5,y<0.5的框提取出来；
                也就是j,k,l,m，在选取gij(也就是标签框分配给的网格的时候)对这四个部分的框都做一个偏移(减去上面的off),也就是下面的gij = (gxy - offsets).long()操作；
                再将这四个部分的框与原始的gxy拼接在一起，总共就是五个部分；
                也就是说：①将每个网格按照2x2分成四个部分，每个部分的框不仅采用当前网格的anchor进行回归，也采用该部分相邻的两个网格的anchor进行回归；
                原yolov3就仅仅采用当前网格的anchor进行回归；
                估计是用来缓解网格效应，但由于v5没发论文，所以也只是推测，yolov4也有相关解决网格效应的措施，是通过对sigmoid输出乘以一个大于1的系数；
                这也与yolov5新的边框回归公式相关；
                由于①，所以中心点回归也从yolov3的0~1的范围变成-0.5~1.5的范围；
                所以中心点回归的公式变为：
                xy.sigmoid() * 2. - 0.5 + cx

                每个标签框采用了更多的anchor来回归，笔者亲测这个操作是增加了召回率，但准确率有所下降
                总体来说map会有所增加
                """
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j.shape = (5, M)
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # t.shape = (5, M, 7)
                # 得到筛选的框(N, 7), N为筛选后的个数
                t = t.repeat((5, 1, 1))[j]
                # 添加偏移量
                # (1, M, 2) + (5, 1, 2) = (5, M, 2) --[j]--> (N, 2)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # b为batch中哪一张图片的索引，c为类别
            b, c = t[:, :2].long().T  # image, class
            # 中心点回归标签
            gxy = t[:, 2:4]  # grid xy
            # 长宽回归标签
            gwh = t[:, 4:6]  # grid wh
            # 对应于原yolov3中，gij = gxy.long()
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            # a为anchor的索引
            a = t[:, 6].long()  # anchor indices
            # 添加索引，方便计算损失的时候取出对应位置的输出
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
