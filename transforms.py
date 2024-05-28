import numpy as np
from PIL import Image,ImageFilter
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2



def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,point=None,sam_mask = None):
        for t in self.transforms:
            image, target, point,sam_mask = t(image, target,point,sam_mask)
        ret = [image,target]

        if point is not None:
            ret.append(point)
        
        if sam_mask is not None:
            ret.append(sam_mask)
        return tuple(ret)

class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target,point=None,sam_mask=None):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, (self.h, self.w), interpolation=Image.NEAREST)
        return image, target,point,sam_mask


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target,point=None,sam_mask=None):
        size = random.randint(self.min_size, self.max_size)  # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
        image = F.resize(image, size)
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, size, interpolation=Image.NEAREST)
        if sam_mask is not None:
            sam_mask = F.resize(sam_mask, size, interpolation=Image.NEAREST)
        return image, target,point,sam_mask


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, target,point=None,sam_mask=None):
        if random.random() < self.p:
            h,w = target.shape
            image = F.hflip(image)
            target = F.hflip(target)
            if point is not None:
                point[0] = w-point[0]-1
            if sam_mask is not None:
                sam_mask = F.hflip(sam_mask)
        return image, target, point, sam_mask


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target,point=None,sam_mask=None):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target, point


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target,point=None):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target, point


class ToTensor(object):
    def __call__(self, image, target,point=None,sam_mask=None):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        if sam_mask is not None:
            sam_mask = torch.as_tensor(np.asarray(sam_mask).copy(), dtype=torch.int64)
        return image, target, point, sam_mask


class RandomAffine(object):
    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, image, target,point=None):
        affine_params = T.RandomAffine.get_params(self.angle, self.translate, self.scale, self.shear, image.size)
        image = F.affine(image, *affine_params)
        target = F.affine(target, *affine_params)
        return image, target, point


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target,point=None,sam_mask=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, point,sam_mask


class RandomGaussianBlur(object):
    def __init__(self,min_sigma=0.1, max_sigma=2.0,p=0.5):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p = p 
    def __call__(self, image, target,point=None,sam_mask=None):
        if random.random() < self.p:
            # sigma = np.random.uniform(self.min_sigma, self.max_sigma)
            sigma = 1
            to_tensor = T.ToTensor()
            to_pil = T.ToPILImage()
            image = to_pil(image)
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
            image = to_tensor(image)
            # image = F.gaussian_blur(image,kernel_size=3,sigma=sigma)

        return image, target, point,sam_mask

class RandomColorJitter(object):
    def __init__(self,brightness, contrast, saturation,hue, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p 

    def __call__(self, image, target,point=None,sam_mask=None):
        if random.random() < self.p:
            color_jit = T.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
            image = color_jit(image)
        return image, target, point,sam_mask