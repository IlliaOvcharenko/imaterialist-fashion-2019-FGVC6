import cv2
import torch
import torchvision

import pandas as pd
import numpy as np

from PIL import Image

from pytorch_toolbelt.utils.rle import rle_encode
from pytorch_toolbelt.utils.rle import rle_decode


class FashionDataset(torch.utils.data.Dataset):
    
    def __init__(self, mode, df, folder, transfrom=None, num_samples=None):
        self.mode = mode
        
        self.df = df
        self.names = df.ImageId.unique().tolist()
        np.random.shuffle(self.names)
        
        self.folder = folder
        self.transfrom = transfrom
        self.num_samples = num_samples
            
    def rle_to_mask(self, item, shape):
        
        mask = np.zeros((46, *shape), dtype=int)
        for i in range(0, 46):
            channel = item[item.ClassId == i]
                
            for j (_, obj) in enumerate(channel.iterrows()):
                mask[i] = mask[i] + (rle_decode(obj.EncodedPixels, shape, int) * (j+1)) 
        mask = mask.astype(np.uint8)
        mask = np.transpose(mask,(1, 2, 0))
        return mask
        
    def __getitem__(self, idx):
        if self.num_samples is not None and self.mode in ["train"]:
            name = np.random.choice(self.names)
        else:
            name = self.names[idx]
        
        origin_fn = self.folder / name
    
        origin  = cv2.imread(str(origin_fn))
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        
        mask = None
        
        if self.mode in ["train", "val"]:
            origin_size = origin.shape[:2]
            item = self.df[self.df.ImageId == name]
            mask = self.rle_to_mask(item, origin_size)
        

        if self.transfrom is not None:
            transformed = self.transfrom(
                image=origin,
                mask=mask
            )
            origin = transformed["image"]
            mask = transformed["mask"]

        
        if mask is None:
            return  name, origin
        return  name, origin, mask

        
    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.names)