import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np
import imageio

from torch.autograd import Variable
from torch import optim
from model import G12, G21
from model import D1, D2


class Solver(object):
    def __init__(self, config, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.use_reconst_loss = config.use_reconst_loss
        self.use_labels = config.use_labels
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.build_model()
        
    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g12 = G12(conv_dim=self.g_conv_dim)
        self.g21 = G21(conv_dim=self.g_conv_dim)
        self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        
        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())
        
        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()
    
    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([4, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)
    
    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)
        iter_per_epoch = min(len(source_iter), len(target_iter))
        
        # fixed target and source for sampling
        fixed_source = self.to_var(source_iter.next()[0])
        fixed_target = self.to_var(target_iter.next()[0])
        
        # loss if use_labels = True
        criterion = nn.CrossEntropyLoss()
        
        for step in range(self.train_iters+1):
            # reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                target_iter = iter(self.target_loader)
                source_iter = iter(self.source_loader)
            
            # load source and target dataset
            source,_,_,_, s_labels = source_iter.next() 
            source, s_labels = self.to_var(source), self.to_var(s_labels).long().squeeze()
            target,_,_,_, m_labels = target_iter.next() 
            target, m_labels = self.to_var(target), self.to_var(m_labels)

            if self.use_labels:
                target_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes]*source.size(0)).long())
                source_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes]*target.size(0)).long())
            
            #============ train D ============#
            
            # train with real images
            self.reset_grad()
            out = self.d1(target)
            if self.use_labels: # False
                d1_loss = criterion(out, m_labels)
            else:
                d1_loss = torch.mean((out-1)**2) # |output-allones|^2
            
            out = self.d2(source)
            if self.use_labels:
                d2_loss = criterion(out, s_labels)
            else:
                d2_loss = torch.mean((out-1)**2)
            
            d_target_loss = d1_loss
            d_source_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()
            
            # train with fake images
            self.reset_grad()
            fake_source = self.g12(target)
            out = self.d2(fake_source)
            if self.use_labels:
                d2_loss = criterion(out, source_fake_labels)
            else:
                d2_loss = torch.mean(out**2) # |output-0|^2
            
            fake_target = self.g21(source)
            out = self.d1(fake_target)
            if self.use_labels:
                d1_loss = criterion(out, target_fake_labels)
            else:
                d1_loss = torch.mean(out**2)
            
            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()
            
            #============ train G ============#
            
            # train target-source-target cycle
            self.reset_grad()
            fake_source = self.g12(target)
            out = self.d2(fake_source)
            reconst_target = self.g21(fake_source)
            if self.use_labels:
                g_loss = criterion(out, m_labels) 
            else:
                g_loss = torch.mean((out-1)**2) 

            if self.use_reconst_loss:
                g_loss += torch.mean((target - reconst_target)**2)

            g_loss.backward()
            self.g_optimizer.step()

            # train source-target-source cycle
            self.reset_grad()
            fake_target = self.g21(source)
            out = self.d1(fake_target)
            reconst_source = self.g12(fake_target)
            if self.use_labels:
                g_loss = criterion(out, s_labels) 
            else:
                g_loss = torch.mean((out-1)**2) 

            if self.use_reconst_loss:
                g_loss += torch.mean((source - reconst_source)**2)

            g_loss.backward()
            self.g_optimizer.step()
            
            # print the log info
            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_target_loss: %.4f, d_source_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f' 
                      %(step+1, self.train_iters, d_real_loss.data, d_target_loss.data, 
                        d_source_loss.data, d_fake_loss.data, g_loss.data))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                fake_source = self.g12(fixed_target)
                fake_target = self.g21(fixed_source)
                
                target, fake_target = self.to_data(fixed_target), self.to_data(fake_target)
                source , fake_source = self.to_data(fixed_source), self.to_data(fake_source)
                
                merged = self.merge_images(target, fake_source)
                path = os.path.join(self.sample_path, 'sample-%d-t-s.png' %(step+1))
                imageio.imwrite(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(source, fake_target)
                path = os.path.join(self.sample_path, 'sample-%d-s-t.png' %(step+1))
                imageio.imwrite(path, merged)
                print ('saved %s' %path)
            
            if (step+1) % 2500 == 0:
                # save the model parameters for each epoch
                g12_path = os.path.join(self.model_path, 'g12-%d.pkl' %(step+1))
                g21_path = os.path.join(self.model_path, 'g21-%d.pkl' %(step+1))
                d1_path = os.path.join(self.model_path, 'd1-%d.pkl' %(step+1))
                d2_path = os.path.join(self.model_path, 'd2-%d.pkl' %(step+1))
                torch.save(self.g12.state_dict(), g12_path)
                torch.save(self.g21.state_dict(), g21_path)
                torch.save(self.d1.state_dict(), d1_path)
                torch.save(self.d2.state_dict(), d2_path)
                
    def gst(self, dataloader):
        iter_ = iter(dataloader)
        
        data_ = self.to_var(iter_.next()[0])
        self.g21.eval()
        fake_ = self.g21(data_) # fake target
        return self.to_data(fake_)


