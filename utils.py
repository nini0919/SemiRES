from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import errno

import os
import os.path as osp

import sys

import matplotlib.pyplot as plt
import numpy as np
import cv2 

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
                sys.stdout.flush()

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))

    def log_semi(self,iterable_zip,print_freq,length,header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(length))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj1,obj2 in iterable_zip:
            data_time.update(time.time() - end)
            yield obj1,obj2
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (length - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, length, eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
                sys.stdout.flush()

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))
        
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    args.local_rank = dist.get_rank()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.barrier()
    setup_for_distributed(is_main_process())

    if args.output_dir:
        mkdir(args.output_dir)
    if args.model_id:
        mkdir(os.path.join('./models/', args.model_id))


def unnormalize(img):
    return img * np.array([[[0.229,0.224,0.225]]])+np.array([[[0.485,0.456,0.406]]])

def rev_normalize_tensor(img):
    _std = torch.tensor([0.229,0.224,0.225]).reshape(1,3,1,1).cuda()
    _mean = torch.tensor([0.485,0.456,0.406]).reshape(1,3,1,1).cuda()
    return img * _std + _mean


def write_img(save_dir,cls,img_list):
    for idx,img in enumerate(img_list):
        plt.imsave(osp.join(save_dir,'{0}_{1}.png'.format(cls,idx)),np.clip(unnormalize(img),0,1))

def write_mask(save_dir,cls,img_list):
    for idx,img in enumerate(img_list):
        plt.imsave(osp.join(save_dir,'{0}_{1}.png'.format(cls,idx)),img,cmap='gray')

def write_npy(save_dir,cls,arr):
    np.save(osp.join(save_dir,'{0}.npy'.format(cls)),arr)


def vis_img(save_dir,img_list):
    for idx,img in enumerate(img_list):
        plt.subplot(1,len(img_list),idx+1)
        plt.imshow(unnormalize(img))
    plt.savefig(save_dir,dpi=600)

def vis_mask(save_dir,img_list):
    for idx,img in enumerate(img_list):
        plt.subplot(1,len(img_list),idx+1)
        plt.imshow(img)
    plt.savefig(save_dir,dpi=600)

def vis_mask_tensor(save_dir,img_tensor):
    for idx in range(len(img_tensor)):
        plt.subplot(1,len(img_tensor),idx+1)
        plt.imshow(img_tensor[idx])
    plt.savefig(save_dir,dpi=600)

def vis_img_mask(save_dir,img,img_w,img_s,target,target_w,target_s):
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.subplot(2,3,2)
    plt.imshow(img_w)
    plt.subplot(2,3,3)
    plt.imshow(img_s)
    plt.subplot(2,3,4)
    plt.imshow(target)
    plt.subplot(2,3,5)
    plt.imshow(target_w)
    plt.subplot(2,3,6)
    plt.imshow(target_s)

    plt.savefig(save_dir,dpi=600)


def show_mask(mask, ax,color=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    if color is not None:
        color = color
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def tag_masks(masks):
    masks = masks.cpu().numpy()
    H,W = masks[0].shape
    ret_mask = np.zeros((H,W),dtype=np.int64)
    for idx,m in enumerate(masks):
        ret_mask[m>0]=idx+1
    return ret_mask
