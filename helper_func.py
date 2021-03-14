import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
# from torch.autograd import Functions
import torch.utils.data as data
# import matplotlib.pyplot as plt
# %matplotlib inline
import pickle
from scipy.signal import butter,lfilter,freqz,detrend,stft
# from scipy.fft import fft
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
# seed = 0
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ['PYTHONHASHSEED'] = str(seed)

def MinMaxNorm(x_train):
    # Normalization function
    x_max = np.max(x_train)
    x_min = np.min(x_train)
    x_train = (x_train-x_min)/(x_max - x_min)
    return x_train

def load_feature(filePath,v_num,b1b2):
    # Load feature and four-channel acceleration data
    labels = np.genfromtxt(filePath+'label.csv', delimiter=',', dtype='f')
    data1 = np.genfromtxt(filePath+'1.csv', delimiter=',', dtype='f').T
    data2 = np.genfromtxt(filePath+'2.csv', delimiter=',', dtype='f').T
    data3 = np.genfromtxt(filePath+'3.csv', delimiter=',', dtype='f').T
    data4 = np.genfromtxt(filePath+'4.csv', delimiter=',', dtype='f').T

    # Signal length
    leng = 4000
    n_sample = 480
    # Calculate STFT
    f,t,Zxx = stft(data1[1,:],nperseg=128)
    fea_bw=np.zeros((3600,np.shape(Zxx)[0],\
                    np.shape(Zxx)[1]))
    fea_fw=np.zeros((3600,np.shape(Zxx)[0],\
                    np.shape(Zxx)[1]))
    fea_bc=np.zeros((3600,np.shape(Zxx)[0],\
                    np.shape(Zxx)[1]))
    fea_fc=np.zeros((3600,np.shape(Zxx)[0],\
                    np.shape(Zxx)[1]))
    for i in range(3600):
        _,_,tmp=stft(data1[i,:],nperseg=128)
        fea_bw[i,:,:] = MinMaxNorm(abs(tmp))
        _,_,tmp=stft(data2[i,:],nperseg=128)
        fea_fw[i,:,:] = MinMaxNorm(abs(tmp))
        _,_,tmp=stft(data3[i,:],nperseg=128)
        fea_bc[i,:,:] = MinMaxNorm(abs(tmp))
        _,_,tmp=stft(data4[i,:],nperseg=128)
        fea_fc[i,:,:] = MinMaxNorm(abs(tmp))
    x_dann = np.zeros((n_sample*2,np.shape(Zxx)[0],\
                    np.shape(Zxx)[1],4))

    feas = [fea_bw,fea_fw,fea_bc,fea_fc]
    n=0
    for fea_ in feas:
        b1vnl2 = (np.floor(labels/100)==1200+v_num)&(labels%(1200+v_num)>0)
        b1vnl4 = (np.floor(labels/100)==1400+v_num)
        b1vnl6 = (np.floor(labels/100)==1600+v_num)&(labels%(1600+v_num)>0)  
        idx_tmp1 = np.logical_or.reduce((b1vnl2,b1vnl4,b1vnl6))
        feas_bb1 = fea_[idx_tmp1]
        b2vnl2 = (np.floor(labels/100)==2200+v_num)&(labels%(2200+v_num)>0)
        b2vnl4 = (np.floor(labels/100)==2400+v_num)
        b2vnl6 = (np.floor(labels/100)==2600+v_num)&(labels%(2600+v_num)>0)
        idx_tmp2 = np.logical_or.reduce((b2vnl2,b2vnl4,b2vnl6))
        feas_bb2 = fea_[idx_tmp2]
        
        feas_bb1_0 = feas_bb1[120:150,:,:]
        feas_bb2_0 = feas_bb2[120:150,:,:]
        if b1b2:
            feas_bb = np.concatenate((feas_bb1,feas_bb1_0,feas_bb1_0,feas_bb1_0,\
                                    feas_bb2,feas_bb2_0,feas_bb2_0,feas_bb2_0))
        else:
            feas_bb = np.concatenate((feas_bb2,feas_bb2_0,feas_bb2_0,feas_bb2_0,\
                                    feas_bb1,feas_bb1_0,feas_bb1_0,feas_bb1_0))     
        
        x_dann[:,:,:,n]=feas_bb
        n = n+1
        idx_tmp = np.logical_or.reduce((idx_tmp1,idx_tmp2))
        label_bb = labels[idx_tmp]

    # damage location labels
    label_l = np.zeros((n_sample,1))
    # damage severity labels
    label_s = np.zeros((n_sample,1))
    damage_ls = np.zeros((150,1))
    for i in range(0,5,1):
        damage_ls[30*i:30*(i+1),0]=i
    label_s[0:120,0] = damage_ls[30:].squeeze()
    label_s[120:270,0] = damage_ls.squeeze()
    label_s[270:390,0] = damage_ls[30:].squeeze()
    label_l[0:120,0] = 1
    label_l[120:150,0] = 0
    label_l[150:270,0] = 2
    label_l[270:390,0] = 3
    label_s = label_s.squeeze()
    label_l = label_l.squeeze()
    # damage detection labels
    label_d = label_l.copy()
    label_d[label_d!=0] = 1
    label_flatten = np.zeros((n_sample,1))
    for i in range(0,13,1):
        label_flatten[i*30:(i+1)*30,0]=i

    return x_dann, label_l, label_s, label_d, label_flatten, label_bb, damage_ls

def test(dataset, model, criterion):
    model.eval()
    test_loss1 = 0
    correct = 0
    pred_l = []
    targets = []

    with torch.no_grad():
        # data is the 4 channel spectrogram; 
        # target_l, target_s and target_d are damage localization, 
        # quantification and detection labels;
        # target are the flatten labels
        for data, target_l, target_s, target_d, target in dataset:
            data, target_l, target_s, target_d, target = data.to('cuda'), target_l.to('cuda'), target_s.to('cuda'), target_d.to('cuda'), target.to('cuda')
            _, output, _ = model(data, 0.)
            test_loss1 += float(criterion(output, target))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_l = np.append(pred_l, pred.cpu().numpy().squeeze())
            targets = np.append(targets, target.cpu().numpy().squeeze())
    # pred_l = pred_l[targets != 0]
    # targets = targets[targets != 0]
    correct = float(np.sum(pred_l == targets))
    test_loss1 /= len(dataset)
    return correct, test_loss1

def test_epoch(epoch, report_itv, source_train, source_test, target_train, target_test, model, criterion):
    losses_save = []
    accuracy_save = []

    correct, test_loss = test(source_train, model, criterion)
    losses_save.append(test_loss)
    accuracy_save.append(correct)
    if epoch % report_itv == 0:
        print('Source Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss, correct))    
    
    correct, test_loss = test(source_test, model, criterion)
    losses_save.append(test_loss)
    accuracy_save.append(correct)
    if epoch % report_itv == 0:
        print('Source test set: Average loss: {:.4f}'.format(
            test_loss))    
    
    correct, test_loss = test(target_train, model, criterion)
    losses_save.append(test_loss)
    accuracy_save.append(correct)
    if epoch % report_itv == 0:
        print('Target train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss, correct))  
    
    correct, test_loss = test(target_test, model, criterion)
    losses_save.append(test_loss)
    accuracy_save.append(correct)
    if epoch % report_itv == 0:
        print('Target test set: Average loss: {:.4f}'.format(
            test_loss))
    return losses_save, accuracy_save


def optimizer_scheduler(optimizer, p):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75
    return optimizer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

def lr_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def exp_lr_scheduler(optimizer, epoch, init_lr, lrd, nevals):
    """Implements torch learning reate decay with SGD"""
    lr = init_lr / (1 + nevals*lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
