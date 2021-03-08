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
#        self.lc = nn.Sequential(
#            nn.Linear(50*self.height*self.height, 100),
#            nn.BatchNorm1d(100),
#            nn.ReLU(),
#            nn.Dropout(),
#            nn.Linear(100, 100),
#            nn.BatchNorm1d(100),
#            nn.ReLU(),
#            nn.Dropout(),
#            nn.Linear(100, 4),
#            nn.Sigmoid(),
#        )
        self.sc = nn.Sequential(
            nn.Linear(50*self.height*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 13),
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
        latent = self.f(x)
        latent = latent.view(-1, 50*self.height*self.height)
        s = self.sc(latent)
        y = GRL.apply(latent, alpha)
        d = self.dc(y)
        s=s.view(s.shape[0],-1)
        d=d.view(d.shape[0],-1)
        return d, s, latent
