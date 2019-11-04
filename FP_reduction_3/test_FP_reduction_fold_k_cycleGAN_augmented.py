#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:18:32 2019

@author: se14
"""
# train the FP reduction network for fold k (user inputted variable)
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
#get_ipython().run_line_magic('matplotlib', 'inline')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dType = torch.float32

# get the current fold from the command line input. 
# for fold k, we use the kth subset as the test, and train on the remaining data
try:
    fold_k = int(sys.argv[1])
    print(f'Testing fold {fold_k}')
except:
    print('Defaulting the fold to 0')
    fold_k = 0

results_filename = f'test_results_fold{fold_k}.csv'

#%% paths
cand_path = '/media/se14/DATA_LACIE/LUNA16/candidates/'
out_path = f'augmented/results_fold_{fold_k}/'
if (not os.path.exists(out_path)) & (out_path != ""): 
    os.makedirs(out_path)

test_subset_folders = [f'subset{i}/' for i in [x for x in range(10) if x==fold_k]]
test_subset_folders = [cand_path + test_subset_folders[i] for i in range(len(test_subset_folders))]

results_filename = out_path + results_filename

if os.path.exists(results_filename):
    print('Results already exist, exiting...')
    sys.exit(1)

#%%
def getParams(model):
    a = list(model.parameters())
    b = [a[i].detach().cpu().numpy() for i in range(len(a))]
    c = [b[i].flatten() for i in range(len(b))]
    d = np.hstack(c)

    return d

def conv3dBasic(ni, nf, ks, stride,padding = 0):
    return nn.Sequential(
            nn.Conv3d(ni, nf, kernel_size = (ks, ks, ks), bias = True, stride = stride, padding = padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(nf))
    
class discriminatorNet_archi_1(nn.Module):
    def __init__(self):
        super().__init__()
        
        # modules
        self.C1 = conv3dBasic(1, 64, 5, 1, 2)
        self.C2 = conv3dBasic(64, 64, 5, 1, 2)
        self.C3 = conv3dBasic(64, 64, 5, 1, 2)
        self.D1 = conv3dBasic(64, 64, 5, 2, 2) #  downsample
        self.FC1 = nn.Linear(64000, 150)
        self.FC2 = nn.Linear(150,1)
        
    def forward(self, x):
        
#        print(x.shape)
        x = self.C1(x)
#        print(x.shape)
        x = self.C2(x)
#        print(x.shape)
        x = self.C3(x)
#        print(x.shape)
        x = self.D1(x)
#        print(x.shape)
#        print(x.view(x.shape[0],-1).shape)
        x = self.FC1(x.view(x.shape[0],-1))
#        print(x.shape)
        x = self.FC2(x)
#        print(x.shape)
        x = torch.sigmoid(x)

        return x
    
#% load the model to be tested
modelToUse_1 = 'augmented/' + f'results_fold_{fold_k}_archi_1/' + 'current_model.pt'
model_1 = discriminatorNet_archi_1()
model_1.load_state_dict(torch.load(modelToUse_1))
model_1 = model_1.to(device)
model_1 = model_1.eval()

#%%
class discriminatorNet_archi_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # modules
        self.C1 = conv3dBasic(1, 64, 5, 1, 2)
        self.M1 = nn.MaxPool3d(2)
        self.C2 = conv3dBasic(64, 64, 5, 1, 2)
        self.C3 = conv3dBasic(64, 64, 5, 1, 2)
        self.D1 = conv3dBasic(64, 64, 5, 2, 2) #  downsample
        self.FC1 = nn.Linear(32768, 250)
        self.FC2 = nn.Linear(250,1)
        
    def forward(self, x):
        
#        print(x.shape)
        x = self.C1(x)
#        print(x.shape)
        x = self.M1(x)
#        print(x.shape)
        x = self.C2(x)
#        print(x.shape)
        x = self.C3(x)
#        print(x.shape)
        x = self.D1(x)
#        print(x.shape)
#        print(x.view(x.shape[0],-1).shape)
        x = self.FC1(x.view(x.shape[0],-1))
#        print(x.shape)
        x = self.FC2(x)
#        print(x.shape)
        x = torch.sigmoid(x)

        return x

#% load the model to be tested
modelToUse_2 = 'augmented/' + f'results_fold_{fold_k}_archi_2/' + 'current_model.pt'
model_2 = discriminatorNet_archi_2()
model_2.load_state_dict(torch.load(modelToUse_2))
model_2 = model_2.to(device)
model_2 = model_2.eval()

#%%
class discriminatorNet_archi_3(nn.Module):
    def __init__(self):
        super().__init__()
        
        # modules
        self.C1 = conv3dBasic(1, 64, 5, 1, 2)
        self.M1 = nn.MaxPool3d(2)
        self.C2 = conv3dBasic(64, 64, 5, 1, 2)
        self.C3 = conv3dBasic(64, 64, 5, 1, 2)
        self.D1 = conv3dBasic(64, 64, 5, 2, 2) #  downsample
        self.FC1 = nn.Linear(64000, 250)
        self.FC2 = nn.Linear(250,1)
        
    def forward(self, x):
        
#        print(x.shape)
        x = self.C1(x)
#        print(x.shape)
        x = self.M1(x)
#        print(x.shape)
        x = self.C2(x)
#        print(x.shape)
        x = self.C3(x)
#        print(x.shape)
        x = self.D1(x)
#        print(x.shape)
#        print(x.view(x.shape[0],-1).shape)
        x = self.FC1(x.view(x.shape[0],-1))
#        print(x.shape)
        x = self.FC2(x)
#        print(x.shape)
        x = torch.sigmoid(x)

        return x
    
#% load the model to be tested
modelToUse_3 = 'augmented/' + f'results_fold_{fold_k}_archi_3/' + 'current_model.pt'
model_3 = discriminatorNet_archi_3()
model_3.load_state_dict(torch.load(modelToUse_3))
model_3 = model_3.to(device)
model_3 = model_3.eval()
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
            csvfiles = f'cand_df_{fldr[-2]}.csv'
#            csvfiles = [f for f in os.listdir(fldr) if os.path.isfile(os.path.join(fldr, f)) if '.csv' in f][0]
            
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
        currDf = self.cand_df.iloc[idx]
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
            
            # crop out a 40 x 40 x 40 shifted region
            # random translation of up to 5 mm in each direction
            transFact = np.round(10*np.random.rand(3)).astype('int16') - 5
            
        elif self.augmentFlag == False:
            transFact = np.array([0,0,0])
            
        currPatch = currPatch[20+transFact[0]:60+transFact[0],
                              20+transFact[1]:60+transFact[1],
                              20+transFact[2]:60+transFact[2]]
        
        currPatch1 = torch.from_numpy(currPatch[10:-10,10:-10,10:-10][None,:,:,:])
        currPatch2 = torch.from_numpy(currPatch[5:-5,5:-5,5:-5][None,:,:,:])
        currPatch3 = torch.from_numpy(currPatch[None,:,:,:])

        # output results
        currPatch = torch.from_numpy(currPatch[None,:,:,:]).to(dtype=dType)
        currLabel = torch.from_numpy(np.array(currLabel)).to(dtype=dType)
        sample = {'image1': currPatch1, 
                  'image2': currPatch2, 
                  'image3': currPatch3, 
                  'labels': currLabel,
                  'seriesuid': currDf['seriesuid'],
                  'coordX': currDf['coordX'],
                  'coordY': currDf['coordY'],
                  'coordZ': currDf['coordZ'],
                  'class' : currDf['class']} # return these values
        
        return sample
    

#%% set up dataloader
batch_size = 12
testData = lidcCandidateLoader(test_subset_folders,augmentFlag=False,balanceFlag=False)
test_dataloader = DataLoader(testData, batch_size = batch_size,shuffle = False,num_workers = 2,pin_memory=True)

#%% perform the test
results_df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','probability'])
with torch.no_grad():
    print('Starting the loop')

    for i, data in enumerate(test_dataloader, 0):
        print(f'{i} of {len(test_dataloader)}')
        
        if np.mod(i,10) == 0:
            time.sleep(0.1) # try to stop overworked cpu by pausing

        inputs1, inputs2, inputs3, labels = data['image1'],data['image2'],data['image3'],data['labels']
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)
        labels = labels.to(device)
                
        outputs_1 = model_1(inputs1).cpu().numpy()
        outputs_2 = model_2(inputs2).cpu().numpy()
        outputs_3 = model_3(inputs3).cpu().numpy()
        
        outputs = 0.3 * outputs_1 + 0.4 * outputs_2 + 0.3 * outputs_3
        
        tmpDf = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','probability'])
        
        for ii in tmpDf.columns[0:-1]:
            tmpDf[ii] = data[ii]
                    
        tmpDf['probability'] = outputs
        
        results_df = results_df.append(tmpDf,ignore_index=False,sort=False).reset_index(drop=True)

results_df.to_csv(results_filename,index=False)