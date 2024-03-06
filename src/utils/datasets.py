from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import glob

class FinetuneDataset(Dataset):
    def __init__(self, dir, qp_path=None, transform=None):
        self.dir = dir
        if qp_path is None:
            self.qps = np.loadtxt(os.path.join(dir, 'qps.txt')).astype(int)
        else:
            self.qps = np.loadtxt(os.path.join(dir, qp_path)).astype(int)
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(self.dir, '*.png')))
    
    def __len__(self):
        return len(self.qps)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.qps[idx]
    
class FileDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(self.dir, '*.png')))
    
    def __len__(self):
        return len(glob.glob(os.path.join(self.dir, '*.png')))
                   
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, idx
