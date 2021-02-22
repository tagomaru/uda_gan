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
import os
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
import pickle
from scipy.signal import butter,lfilter,freqz,detrend,stft
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from GRL import GRL

class Dann(nn.Module):
    def __init__(self):
        super(Dann, self).__init__()
        self.height = 5
        self.f = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=5),
                nn.BatchNorm2d(64),
#                 nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(64, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=3),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
        self.lc = nn.Sequential(
            nn.Linear(50*self.height*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 4),
            nn.Sigmoid(),
        )
        self.sc = nn.Sequential(
            nn.Linear(50*self.height*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 5),
            nn.Sigmoid(),
        )
        self.dc = nn.Sequential(
            nn.Linear(50*self.height*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Sigmoid(),
        )
    def forward(self, x, alpha):
#         print(x.shape)
        xl,xq = x
        latent_xl = self.f(xl)
        latent_xq = self.f(xq)
#         print(x.shape)
        latent_xl = latent_xl.view(-1, 50*self.height*self.height)
        latent_xq = latent_xq.view(-1, 50*self.height*self.height)
        x = self.lc(latent_xl)
        s = self.sc(latent_xq)
        y_xl = GRL.apply(latent_xl, alpha)
        d_xl = self.dc(y_xl)
        y_xq = GRL.apply(latent_xq, alpha)
        d_xq = self.dc(y_xq)
        x=x.view(x.shape[0],-1)
        s=s.view(s.shape[0],-1)
        d_xl=d_xl.view(d_xl.shape[0],-1)
        d_xq=d_xq.view(d_xq.shape[0],-1)
        return x, (d_xl,d_xq), s,(latent_xl,latent_xq)