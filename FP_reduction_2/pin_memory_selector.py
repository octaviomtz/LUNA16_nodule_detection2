#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:07:08 2019

@author: se14
"""
# test various values of pin_memory to find best for speed
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
import pandas as pd
import scipy.sparse
import scipy.ndimage
import os
import warnings
import sklearn.metrics
import SimpleITK as sitk
import math

warnings.filterwarnings('ignore', '.*output shape of zoom.*')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
np.random.seed(0)
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
get_ipython().run_line_magic('matplotlib', 'auto')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dType = torch.float32

# get the current fold from the command line input. 
# for fold k, we use the kth subset as the test, and train on the remaining data
fold_k = 0#int(sys.argv[1])
print(f'Training fold {fold_k}')

#%% paths
cand_path = '/media/se14/DATA/LUNA16/candidates/'

train_subset_folders = [f'subset{i}/' for i in [x for x in range(10) if x!=fold_k]]
train_subset_folders = [cand_path + train_subset_folders[i] for i in range(len(train_subset_folders))]

# set the validation subset
val_subset_folders = [train_subset_folders[fold_k-1]]

# and then remove this from the training subsets
train_subset_folders.remove(val_subset_folders[0]) 

#%% network architecture for FP reduction
def conv2dBasic(ni, nf, ks, stride,padding = 0):
    return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size = (ks, ks), bias = True, stride = stride, padding = padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nf))
    
class cnn2d(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = conv2dBasic(1, 24, 5, stride = 1, padding = 2)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = conv2dBasic(24, 32, 3, stride = 1, padding = 0)
        self.mp2 = nn.MaxPool2d(2)
        self.conv3 = conv2dBasic(32, 48, 3, stride = 1)
        self.mp3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(48 * 6 * 6, 16)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.fc1(x.view(x.shape[0],-1))
        
        return x 
    
class discriminatorNet(nn.Module):
    def __init__(self):
        super().__init__() 

        # define modules
        self.netView1 = cnn2d()
        self.netView2 = cnn2d()
        self.netView3 = cnn2d()
        self.fc_out = nn.Linear(3 * 16, 1)
        
    def forward(self, x):
        
        # define the forward pass
        x1 = self.netView1(x[:,0,:,:][:,None,:,:])
        x2 = self.netView2(x[:,1,:,:][:,None,:,:])
        x3 = self.netView3(x[:,2,:,:][:,None,:,:])
        
        # concatenate the results
        x = torch.cat((x1,x2,x3),dim=1)
        
        x = self.fc_out(x)
        
        x = torch.sigmoid(x).view(-1)
                        
        return x
    
# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_net(m):
    if (type(m) == nn.Linear) or (type(m) == nn.modules.conv.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        
    if hasattr(m, 'bias'):
        try:
            nn.init.constant_(m.bias,0.0)
        except:
            pass

    
model = discriminatorNet()
model = model.to(dtype=dType)
model = model.apply(init_net).to(device)
#image = torch.zeros((1,1,40,40,40)).to(dtype=dType).to(device)
#out = model(image)
#print(out)


#%% dataset object to read in all candidates from our training data
def eulerAnglesToRotationMatrix(theta):

    thetaRad = np.zeros_like(theta)
    for ii in range(len(theta)):
        thetaRad[ii] = (theta[ii] / 180.) * np.pi     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                     
    R = np.dot(R_z, np.dot( R_y, R_x )) 
    return R

class lidcCandidateLoader(Dataset):
    
    def __init__(self,data_folders,augmentFlag,balanceFlag):
        # data_folders are the locations of the data that we want to use
        # e.g. '/media/se14/DATA/LUNA16/candidates/subset9/'
        cand_df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','diameter_mm','filename'])
        for fldr in data_folders:
            csvfiles = [f for f in os.listdir(fldr) if os.path.isfile(os.path.join(fldr, f)) if '.csv' in f][0]
            
            cand_df = cand_df.append(pd.read_csv(fldr + csvfiles),ignore_index=True,sort=False)
            
        # if we are balancing the data, then we need to do that here by oversampling
        # the positives
        if balanceFlag==True:
            true_df = cand_df.loc[cand_df['class']==1]
            false_df = cand_df.loc[cand_df['class']==0]
            
            numRepeats = int(np.ceil(len(false_df) / len(true_df)))
            true_df_aug = pd.concat([true_df]*numRepeats)[0:len(false_df)]
            
            cand_df = true_df_aug.append(false_df,ignore_index=False,sort=False).reset_index(drop=True)
        
        self.cand_df = cand_df
        
        # only set augmentation for training, not validation or testing
        if augmentFlag == True:
            self.augmentFlag = True
        else:
            self.augmentFlag = False
        
    def __len__(self):
        return len(self.cand_df)
#        return 2
    
    def __getitem__(self,idx):
        currFileName = self.cand_df.iloc[idx]['filename']
        currLabel = self.cand_df.iloc[idx]['class']
        currPatch = np.fromfile(currFileName,dtype='int16').astype('float32')
        currPatch = currPatch.reshape((80,80,80))

        # some intensity transforms/normalisations
        currPatch[np.where(currPatch<-1000)]= -1000
        currPatch[np.where(currPatch>400)] = 400
        currPatch = (currPatch + 1000)/1400
        
        # augment if augmentFlag is True
        if self.augmentFlag == True:  
    
            # random flippings
            flipX = np.random.rand() > 0.5
            flipY = np.random.rand() > 0.5
            if flipX:
                currPatch = np.flip(currPatch,axis=1).copy()
                
            if flipY:
                currPatch = np.flip(currPatch,axis=2).copy()
                
            # random offset
            offSet = 0.3*np.random.rand() - 0.15
            currPatch += offSet
            
            # random Gaussian blur
            randSigma = 0#np.random.rand() # between 0 and 1mm smoothing (standard deviation, not FWHM!)
            currPatch = scipy.ndimage.gaussian_filter(currPatch,randSigma)
            
            # random rotation and scaling
            scaleFact = 0.5*np.random.rand() + 0.75
            rotFactX = 60.*np.random.rand() - 30
            rotFactY = 60.*np.random.rand() - 30
            rotFactZ = 60.*np.random.rand() - 30
                        
            image_center = tuple(np.array(currPatch.shape) / 2 - 0.5)
            
            rotMat = eulerAnglesToRotationMatrix((rotFactX,rotFactY,rotFactZ))
            
            scaleMat = np.eye(3,3)
            scaleMat[0,0] *= scaleFact
            scaleMat[1,1] *= scaleFact
            scaleMat[2,2] *= scaleFact
            
            affMat = np.dot(rotMat,scaleMat)
            
            affine = sitk.AffineTransform(3)
            affine.SetMatrix(affMat.ravel())
            affine.SetCenter(image_center)
            
            img = sitk.GetImageFromArray(currPatch)
            refImg = img
            imgNew = sitk.Resample(img, refImg, affine,sitk.sitkLinear,0)
            
            currPatch = sitk.GetArrayFromImage(imgNew).copy()
            
        currPatch = currPatch[10:70,
                              10:70,
                              10:70]
        
        # stack three orthogonal views for classification
        currPatch = np.stack((currPatch[30,:,:],currPatch[:,30,:],currPatch[:,:,30]))
        
        # output results
        currPatch = torch.from_numpy(currPatch).to(dtype=dType)
        currLabel = torch.from_numpy(np.array(currLabel)).to(dtype=dType)
        sample = {'image': currPatch, 'labels': currLabel, 'candIdx' : idx} # return these values
        
        return sample


#%% set up dataloader
batch_size = 256
trainData = lidcCandidateLoader(train_subset_folders,augmentFlag=True,balanceFlag=True)


#%% set up training
criterion = torch.nn.BCELoss()
LR = 1e-4
optimizer = optim.Adam(model.parameters(),lr = LR)

optimizer.zero_grad()

#%% main loop
num_workers = np.arange(1,21)
all_timings = np.zeros((20,10))
for ii in range(10):
    for _, n in enumerate(num_workers):
        print(n)
        train_dataloader = DataLoader(trainData, batch_size = batch_size,shuffle = True,num_workers = n,pin_memory=True)
    
        start = time.time()
    
        print(f'Starting with num_workers = {n}')
        for i, data in enumerate(train_dataloader, 0):
            print(f'{i} of {len(train_dataloader)}')
            # get the inputs
            inputs, labels = data['image'],data['labels']
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            # forward + backward + optimize (every numAccum iterations)
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # calculate loss
            print(f'Batch loss = {loss.item()}')
        
            loss.backward() # backprop the loss to each weight to get gradients
            
            optimizer.step() # take a step in this direction according to our optimiser
            optimizer.zero_grad()
            
                
        end = time.time()
        all_timings[n-1,ii] = end - start

plt.plot(np.arange(1,21),np.mean(all_timings,axis=1))












