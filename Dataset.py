import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
from torch.autograd import Function
import torchvision
import torch.utils.data as data
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label_l, label_s, label_d, label_f, is_train = True):
        n = len(data)
        self.data = data
        self.label_l = label_l
        self.label_s = label_s
        self.label_d = label_d
        self.label_f = label_f
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    # Get one sample
    def __getitem__(self, index):  
        labels_l = int(self.label_l[index])
        labels_s = int(self.label_s[index])
        labels_d = int(self.label_d[index])
        labels_f = int(self.label_f[index])
        img = self.data[index][0:64,:,:]
        if not self.is_train:
            img = img + np.random.randn(64,64,4) * 0.01
        else:
            img = img
        img = torch.tensor(img).float().transpose(0,2)

        return img, torch.tensor(labels_l).long(), torch.tensor(labels_s).long(), torch.tensor(labels_d).long(), torch.tensor(labels_f).long()
