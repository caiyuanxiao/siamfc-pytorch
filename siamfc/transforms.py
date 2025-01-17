from __future__ import absolute_import, division

import cv2
import numpy as np
import numbers
import torch

from . import ops


__all__ = ['SiamFCTransforms']  #只公开括号里面的模块


class Compose(object):  

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch
    
    def __call__(self, img):
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])   #定义插值
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)  #np.random.uniform（a,b）表示从一个均匀分布[a,b)中随机采样
        out_size = (
            round(img.shape[1] * scale), #img.shape[1]读取image第一列的长度
            round(img.shape[0] * scale))
        return cv2.resize(img, out_size, interpolation=interp)  #使用resize缩放图像，输入为image，输出尺寸outsize，使用插值


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]    #image的行数与列数
        tw, th = self.size
        i = round((h - th) / 2.)
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)   #对image进行填充
            i += npad
            j += npad

        return img[i:i + th, j:j + tw]


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):   #判断size和number是否同一类型
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


class ToTensor(object):

    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2, 0, 1)) ##.permute()将tensor的维度换位。


class SiamFCTransforms(object): 

    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            CenterCrop(exemplar_sz),
            ToTensor()])              #对模板进行转换
        self.transforms_x = Compose([     
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            ToTensor()])             #对样本进行转换
    
    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_sz)
        x = self._crop(x, box_x, self.instance_sz)
        z = self.transforms_z(z)
        x = self.transforms_x(x)
        return z, x              #裁剪后进行转换，得到训练图像对
    
    def _crop(self, img, box, out_size):  #定义裁剪函数
        # convert box to 0-indexed and center b ased [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz

        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = ops.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)
        
        return patch
