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

class GRL(Function):
	# Gradient reverse layer
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None
