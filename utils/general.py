# YOLOv5 general utils

import contextlib
import glob
import logging
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output

import cv2
import math
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils.google_utils import gsutil_getsize
from utils.metrics import box_iou, fitness
from utils.torch_utils import init_torch_seeds

# Settings
# 设置torch，np, pandas的显示限制
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
# 限制opencv-python的线程数 
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
# 设置numpy的线程数
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


class timeout(contextlib.ContextDecorator):
    # Usage: @timeout(seconds) decorator or 'with timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


def set_logging(rank=-1, verbose=True):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)


def init_seeds(seed=0):
    """设置固定随机种子"""
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    """获取最新一次训练下的last.pt，方便resume
    os.path.getctime获取文件的创建时间"""
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def is_docker():
    """是否为docker环境"""
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def is_colab():
    """判断是否为google colab环境"""
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except Exception as e:
        return False


def is_pip():
    """判断文件是否为一个pip包
    Path(__file__).absolute().parts, 将当前文件路径划分为元组
    example：/home/laughing/yolov5/train.py -> ('/', 'home', 'laughing', 'yolov5', 'train.py')"""
    # Is file in a pip package?
    return 'site-packages' in Path(__file__).absolute().parts


def emojis(str=''):
    """这里是按照系统来选取str，使在不同系统都能够打印表情"""
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_size(file):
    """获取文件大小, 单位MB"""
    # Return file size in MB
    return Path(file).stat().st_size / 1e6


def check_online():
    """检查网络是否连接"""
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


def check_git_status(err_msg=', for updates see https://github.com/ultralytics/yolov5'):
    """检查当前代码是否为最新"""
    # Recommend 'git pull' if code is out of date
    # colorstr函数可改变“github:”字体颜色，具体查看colorstr注释
    print(colorstr('github: '), end='')
    try:
        # 判断文件夹是否存在.git文件夹(git管理的文件夹)
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        # 判断是否为docker环境
        assert not is_docker(), 'skipping check (Docker image)'
        # 判断是否联网
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        # check_output:shell命令，并返回结果
        # 将最新的代码拉取到本地，并返回代码地址url
        url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
        # 获取当前分支名branch
        branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
        # 返回落后的提交数
        n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))  # emoji-safe
    except Exception as e:
        print(f'{e}{err_msg}')


def check_python(minimum='3.6.2'):
    """检查当前python版本是否满足要求
    platform.python_version()获取当前python版本"""
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ')


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False):
    """检测当前环境的版本是否满足要求"""
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    # 如果pinned=True则要求环境版本要一致，否则只要大于等于即可
    result = (current == minimum) if pinned else (current >= minimum)
    assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'


def check_requirements(requirements='requirements.txt', exclude=()):
    """检查当前环境是够满足要求
    exclude:不需要检查的环境包"""
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    # 将requirements显示这是为红色
    prefix = colorstr('red', 'bold', 'requirements:')
    # 检查python环境是否满足要求
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        # 不存在requirements则直接返回
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        # x.name为包名，x.specifier为版本要求，例如>=1.0.2
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            # 检查包和其依赖，为安装则报错，然后跳到except中安装
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            print(f"{prefix} {r} not found and is required by YOLOv5, attempting auto-update...")
            try:
                assert check_online(), f"'pip install {r}' skipped (offline)"
                # 安装/更新包
                print(check_output(f"pip install '{r}'", shell=True).decode())
                n += 1
            except Exception as e:
                print(f'{prefix} {e}')

    if n:  # if packages updated
        # 显示安装了多少包
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))  # emoji-safe


def check_img_size(img_size, s=32):
    """检查image大小, 保证img_size能整除s(32)"""
    # Verify img_size is a multiple of stride s
    # 是img_size可以整除s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_imshow():
    """检查当前环境是否满足cv2.imshow"""
    # Check if environment supports image displays
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_file(file):
    """检查file"""
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    # 判断当前文件是否存在，存在则直接返回
    if Path(file).is_file() or file == '':  # exists
        return file
    # 如果file以http或https开头，则自动下载该文件
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)
        # 下载之后判断该文件是否存在，并文件大小大于0
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        # 在当前目录下搜索该文件
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_dataset(data, autodownload=True):
    """检查dataset"""
    # Download dataset if not found locally
    # 获取数据集所在的path, 来自数据集配置文件data.yaml, 没有path返回‘’
    path = Path(data.get('path', ''))  # optional 'path' field
    if path:
        for k in 'train', 'val', 'test':
            if data.get(k):  # prepend path
                # 获取train,val,test数据路径, 与path路径拼接
                data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]
    if val:
        # Path(x).resolve获取x的绝对路径，解析其中的所有符号连接，并规范化
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            # 如果val文件夹不存在，且yaml文件中有download选项，且设置autodownload，则自动下载
            if s and autodownload:  # download script
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} ...')
                    torch.hub.download_url_to_file(s, f)
                    root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    r = os.system(f'unzip -q {f} -d {root} && rm {f}')  # unzip
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # print result
            else:
                raise Exception('Dataset not found.')


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function
    def download_one(url, dir):
        # Download 1 file
        # f文件名
        f = dir / Path(url).name  # filename
        if not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        # f.suffix文件后缀
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                s = f'unzip -qo {f} -d {dir}'  # unzip -quiet -overwrite
            elif f.suffix == '.gz':
                s = f'tar xfz {f} --directory {f.parent}'  # unzip
            if delete:  # delete zip file after unzip
                s += f' && rm {f}'
            os.system(s)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    # 多进程下载
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in tuple(url) if isinstance(url, str) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # 返回能被divisor整除的x
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    """将字符串中的特殊符号(如@，*等)替换为下划线_"""
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """训练时的余弦退火公式"""
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    """改变输出文本的颜色和字体，格式：(颜色，字体，文本)"""
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    """根据labels初始化图片采样权重"""
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    # 将所有label拼接
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    # 统计每个类别的数量, bincount返回一个长度为nc的数组，索引的值为该类别的数量
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    # 替换空的bins为1
    weights[weights == 0] = 1  # replace empty bins with 1
    # 计算采样权重
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """根据class_weights以及配合每张图片包含的类别数更新采样权重"""
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    """coco数据集的索引映射，因为coco数据的索引其实是到91的，但只有80个类，即0~91之间有些索引没有用到"""
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    """xyxy左上角+右下角坐标 -> xywh中心点+宽高"""
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """xywh中心点+宽高 -> xyxy左上角+右下角坐标"""
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """归一化的xywh中心点+宽高 -> xyxy左上角+右下角坐标"""
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """xyxy左上角+右下角坐标 -> 归一化的xywh中心点+宽高"""
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    # 规范xyxy坐标在图片宽高内
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """归一化的segment坐标 -> 基于原图的segment坐标
    在数据增强mosaic中有使用"""
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    """根据segment的坐标，取xy坐标的最大最小值，得到边框的坐标
    处理一个segment"""
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    # 选取在图片内的xy坐标索引
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    """同上，只不多是处理多个segment，并且将xyxy -> xywh"""
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """对segment重新采样，比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)"""
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    # segments = [(n, 2), ] * N
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """基于网络input-size的坐标 -> 基于原图大小的坐标
    img1_shape:网络input-size；
    coord：网络输出坐标；
    img0_shape：原图shape"""
    # Rescale coords (xyxy) from img1_shape to img0_shape
    # 计算input-size到原图大小需要的padding
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    # 坐标加上pad，然后缩放会原图大小
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # 将坐标规范为原图大小以内
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """规范boxes坐标到shape大小以内"""
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """对预测做nms
        prediction:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别, 默认为None则保留全部类别
        agnostic:进行nms是否也去除不同类别之间的框
        multi_label:是否采用多标签
        labels:加入真实标签进行nms,目前是在test.py中保存预测到txt文件时，save_hybrid选项有使用
        max_det:nms之后保留的最多框数量
    """
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # nc类别数，xc:预测中置信度大于阈值的索引
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    # 对置信度阈值与iou阈值进行检查，range [0, 1]
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # 设置最小的wh, 最大的wh，现在代码没有使用到min_wh,只有max_wh在计算nms的时候作为一个偏移量
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    # 选取预测中最多30000个框做nms
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # 限制做nms时间为10s，超时直接退出
    time_limit = 10.0  # seconds to quit after
    # 是否需要冗余的检测结果，仅在merge=True时起作用
    redundant = True  # require redundant detections
    # 多标签,即一个box保留多个类别
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    # prediction为一个batch的预测，下面是对每张图片做处理
    # prediction的shape为[N, M, 5+nc], N为图片数，M为每张图片预测总数
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 选取当前图片的预测
        x = x[xc[xi]]  # confidence

        # 拼接真实label
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # 将预测置信度conf与class预测分数相乘 作为新的置信度
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # 整合预测(n, nc + 5) -> (n, 6)
        if multi_label:
            # 多标签的保留多个预测类别, 大于conf_thres的都保留
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # 仅保留可能性最大的类别预测
            # 找到最大的新置信度与其索引
            conf, j = x[:, 5:].max(1, keepdim=True)
            # 拼接[box预测框, conf置信度, cls预测类别]
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        # 根据参数classes筛选保留的类别
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # 将x按照conf排序，取前max_nms个
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        # 根据类别设置一个每个框的偏移量
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes加上一个偏移量，方便做nms
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # nms
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            # 选取max_det个预测结果
            i = i[:max_det]
        # 融合预测框
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            # 计算iou并取大于iou阈值的索引, 返回的shape：[N, M], N为boxes[i]的个数，M为boxes的个数
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            # 根据score和iou设置权重
            weights = iou * scores[None]  # box weights
            # 融合边框
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                # 只取iou之和大于1的融合边框
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    """在训练时pt文件会保存其他东西，比如优化器，训练结果等，
    strip_optimizer将权重中保存的其他东西去掉，仅保留网络权重"""
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    # 将其他东西设置为None
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    # 模型转为F16
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    # 再保存
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    """
    写入results和对应的hyp到evolve.txt
    evolve.txt文件每一行为一次进化的结果
    一行中前七个数字为(P, R, mAP, F1, test_losses=(box, obj, cls))，之后为hyp
    保存hyp到yaml文件
    """
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    # a超参数的名字，b超参数的值，c模型指标与损失
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    # 保存进化的hyp
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    """再对检测之后的目标进行二次分类
    x：yolov5预测
    model：分类模型
    img：网络输入img
    im0：原图
    """
    # Apply a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            # 将坐标转为xywh格式
            b = xyxy2xywh(d[:, :4])  # boxes
            # 将预测坐标框取wh中长的边作为边长，转为正方形框
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            # 对框进行pad
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            # 再转回xyxy格式
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            # 将坐标转为基于原图坐标
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                # 裁剪box, resize
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)
            # 分类器前向推理
            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            # 保留分类器与检测器分类一致的结果
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    """将预测框截图保存下来
    xyxy:左上角右下角坐标
    im:原图
    gain:对预测框进行resize
    pad:对预测框进行pad
    square:是否保存为正方形
    BGR:当前图片是否为BGR通道
    save：是否保存
    """
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    # 将预测坐标框取wh中长的边作为边长，转为正方形框
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    # 对框进行resize和pad
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    # 转换为xyxy格式并规范坐标在原图长宽以内
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    # 裁剪并保存
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)
    return crop


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """根据文件夹中已有的文件名，自动获得新路径或文件名"""
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    # 使用pathlib处理文件
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        # 获取文件后缀
        suffix = path.suffix
        # 去掉后缀的path
        path = path.with_suffix('')
        # 获取所有以{path}{sep}开头的文件
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # 在dirs中找到以数字结尾的文件
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # 获取dirs文件结尾的数字
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 最大的数字+1
        n = max(i) + 1 if i else 2  # increment number
        # 设置新文件的文件名
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    # 获取文件路径并创建
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
