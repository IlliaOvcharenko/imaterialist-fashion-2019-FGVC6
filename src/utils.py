import torch
import torchvision
import time
import datetime
import math
import random

import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image
from pathlib import Path
from osgeo import gdal

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize(img_tensor):
    img_tensor = img_tensor.clone()
    for t, m, s in zip(img_tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return img_tensor


def image_to_std_tensor(image, **params):
    image = torchvision.transforms.functional.to_tensor(image)
    image = torchvision.transforms.functional.normalize(image, MEAN, STD)
    return image


def mask_to_tensor(mask, **params):
    mask = np.transpose(mask,(2,0,1))
    return torch.tensor(mask).long()

custom_to_std_tensor = A.Lambda(image=image_to_std_tensor, mask=mask_to_tensor)

def mask_lose_objects_info(mask,  **params):
    return (mask > 0).long()

custom_mask_lose_objects_info = A.Lambda(mask=mask_lose_objects_info)

def only_channel(channel):
    def  func(mask, **params):
        return (mask == channel).long()  
    return func

def to_pil(a, **params):
    return Image.fromarray(a)

custom_to_pil = A.Lambda(
    image=to_pil,
    mask=to_pil
)


def get_time_suffix():
    return datetime.datetime.today().strftime("%Y-%m-%d-%Hh-%Mm")


class MaskInfo:
    def __init__(self, mask, r=0.0, b=0.0, g=0.0):
        self.value = mask
        self.r = r
        self.g = g
        self.b = b
        

def blend(origin, *masks, alpha=0.5):
    img = torchvision.transforms.functional.to_pil_image(origin)
    
    for mask in masks:
        if mask is not None and mask.value is not None and mask.value.sum() != 0.0:
            mask = torchvision.transforms.functional.to_pil_image(torch.cat([
                torch.stack([mask.value.float()]) * mask.r,
                torch.stack([mask.value.float()]) * mask.g,
                torch.stack([mask.value.float()]) * mask.b,
            ]))
            img = Image.blend(img, mask, alpha)
    
    return img


class InterpolateWrapper(torch.nn.Module):
    def __init__(self, model, step=32):
        super().__init__()
        
        self.model = model
        self.step = step
        
    def forward(self, x):
        initial_size = list(x.size()[-2:])
        interpolated_size = [(d // self.step) * self.step for d in initial_size] 
        
        x = torch.nn.functional.interpolate(x, interpolated_size)
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, initial_size)
        
        return x
    

def time_wrap(func, *args, desc="execution time:", **kwargs):
    start_time = time.time()
    
    out = func(*args, **kwargs)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print(desc, exec_time)
    
    return out


class time_collection_wrap:
    def __init__(self, collection, desc="item load time:"):
        self.collection_iter = iter(collection)
        self.desc = desc
    
    def __iter__(self):
        return self
    
    def __next__(self):        
        return time_wrap(next, self.collection_iter, desc=self.desc)
    
class ModelEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or np.ones(len(models))
    
    def __call__(self, x):
        res = []
        for model, weight in zip(self.models, self.weights):
            res.append(model(x)*weight)
        res = torch.stack(res)
        return torch.sum(res, dim=0) /sum(self.weights)
    

def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
