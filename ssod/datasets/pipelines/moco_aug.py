import mmcv
import numpy as np
import imgaug.augmenters as iaa
import torch
from torchvision import transforms
from imgaug.augmenters.geometric import Affine
import torchvision

from PIL import ImageFilter, Image
import random

from mmdet.datasets import PIPELINES

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

@PIPELINES.register_module()
class MOCOTransform:
    def __init__(self, mem_pip=False, gray_p=0.2, gaussian_sigma=[.1, 2.], gaussian_p=0.5, color_jitter_p=0.8):
        self.mem_pip = mem_pip
        self.random_gray_scale = torchvision.transforms.RandomGrayscale(p=gray_p)
        self.random_gaussian_blur = torchvision.transforms.RandomApply([GaussianBlur(sigma=gaussian_sigma)], p=gaussian_p)
        self.random_colorjitter = torchvision.transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=color_jitter_p),
    
    def __call__(self, results):
        img = results['img']
        img = Image.fromarray(np.uint8(img))
        img = self.random_colorjitter[0](img)
        img = self.random_gray_scale(img)
        img = self.random_gaussian_blur(img)
        img = np.asarray(img)
        results['img'] = img
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode