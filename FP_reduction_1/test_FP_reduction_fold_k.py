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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dType = torch.float32

# get the current fold from the command line input. 
# for fold k, we use the kth subset as the test, and train on the remaining data
fold_k = (int(sys.argv[1]))
print(f'Testing fold {fold_k}')

results_filename = f'test_results_fold{fold_k}.csv'

#%% paths
cand_path = '/media/se14/DATA_LACIE/LUNA16/candidates/'
out_path = f'results_fold_{fold_k}/'
if (not os.path.exists(out_path)) & (out_path != ""): 
    os.makedirs(out_path)

test_subset_folders = [f'subset{i}/' for i in [x for x in range(10) if x==fold_k]]
test_subset_folders = [cand_path + test_subset_folders[i] for i in range(len(test_subset_folders))]

results_filename = out_path + results_filename

if os.path.exists(results_filename):
    print('Results already exist, exiting...')
    sys.exit(1)

#%% network architecture for FP reduction
# Cheng et al LUNA16 paper
def conv3dBasic(ni, nf, ks, stride,padding = 0):
    return nn.Sequential(
            nn.Conv3d(ni, nf, kernel_size = (ks, ks, ks), bias = True, stride = stride, padding = padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(nf))
    
class discriminatorNet(nn.Module):
    def __init__(self):
        super().__init__() 
        # first block convolutions
        self.block1_L1 = conv3dBasic(1, 24, 3, 2, 1)
        self.block1_L2 = conv3dBasic(24, 32, 3, 1, 1)
        self.block1_R = conv3dBasic(1, 32, 1, 2, 0)
        
        # second block convolutions
        self.block2_L1 = conv3dBasic(32, 48, 3, 1, 1)
        self.block2_L2 = conv3dBasic(48, 48, 3, 1, 1)
        self.block2_R = conv3dBasic(32, 48, 1, 1, 0)
        
        # 3rd block
        self.block3_1 = conv3dBasic(48, 48, 3, 1, 1)
        self.block3_2 = conv3dBasic(48, 48, 3, 1, 1)
        
        # 4th block
        self.block4_L1 = conv3dBasic(48, 64, 3, 2, 1)
        self.block4_L2 = conv3dBasic(64, 64, 3, 1, 1)
        self.block4_R = conv3dBasic(48, 64, 1, 2, 0)
        
        # 5th block
        self.block5_L1 = conv3dBasic(64, 96, 3, 1, 1)
        self.block5_L2 = conv3dBasic(96, 96, 3, 1, 1)
        self.block5_R = conv3dBasic(64, 96, 1, 1, 0)
        
        # 6th block
        self.block6_L1 = conv3dBasic(96, 96, 3, 1, 1)
        self.block6_L2 = conv3dBasic(96, 96, 3, 1, 1)
        self.block6_R = conv3dBasic(96, 96, 1, 1, 0)
        
        # 7th block
        self.block7_1 = conv3dBasic(96, 96, 3, 1, 1)
        self.block7_2 = conv3dBasic(96, 96, 3, 1, 1)
        
        # 8th block
        self.block8_L1 = conv3dBasic(96, 128, 3, 2, 1)
        self.block8_L2 = conv3dBasic(128, 128, 3, 1, 1)
        self.block8_R = conv3dBasic(96, 128, 1, 2, 0)
        
        # 9th block
        self.block9_L1 = conv3dBasic(128, 128, 3, 1, 1)
        self.block9_L2 = conv3dBasic(128, 128, 3, 1, 1)
        self.block9_R = conv3dBasic(128, 128, 1, 1, 0)
        
        # 10th block
        self.block10_1 = conv3dBasic(128, 128, 3, 1, 1)
        self.block10_2 = conv3dBasic(128, 128, 3, 1, 1)
        
        self.testFC = nn.Sequential(nn.Linear(5*5*5*128,128),nn.ReLU(inplace=True))
        
        # 11th block - have a global average pool to give a 128 vector, then we
        # fully connect this to a 2-element softmax
#        self.block11_2 = nn.Linear(128, 2)
        self.block11_2 = nn.Linear(128, 1) # for single-value output
        
        # experimental dropout layers
        self.dropout1 = nn.Dropout3d(p=0.5)
        self.dropout2 = nn.Dropout3d(p=0.6)
        self.dropout3 = nn.Dropout3d(p=0.5)
        
    def forward(self, x):
        
        # 1st block
        xL = self.block1_L1(x)
        xL = self.block1_L2(xL)
        xR = self.block1_R(x)
        x = xL + xR

        # 2nd block
        xL = self.block2_L1(x)
        xL = self.block2_L2(xL)
        xR = self.block2_R(x)
        x = xL + xR

        # 3rd block
        x1 = self.block3_1(x)
        x1 = self.block3_2(x1)
        x = x + x1

        # 4th block
        xL = self.block4_L1(x)
        xL = self.block4_L2(xL)
        xR = self.block4_R(x)
        x = xL + xR

        # 5th block
        xL = self.block5_L1(x)
        xL = self.block5_L2(xL)
        xR = self.block5_R(x)
        x = xL + xR
        
#        # experimental dropout---------
#        x = self.dropout1(x) 
#        #-----------------------------

        # 6th block
        xL = self.block6_L1(x)
        xL = self.block6_L2(xL)
        xR = self.block6_R(x)
        x = xL + xR

        # 7th block
        x1 = self.block7_1(x)
        x1 = self.block7_2(x1)
        x = x + x1     

        # 8th block
        xL = self.block8_L1(x)
        xL = self.block8_L2(xL)
        xR = self.block8_R(x)
        x = xL + xR

        # 9th block
        xL = self.block9_L1(x)
        xL = self.block9_L2(xL)
        xR = self.block9_R(x)
        x = xL + xR

        # 10th block
        x1 = self.block10_1(x)
        x1 = self.block10_2(x1)
        x = x + x1  
        
#        # experimental dropout---------
#        x = self.dropout2(x) 
#        #-----------------------------

        # 11th block
        x = x.view(x.size(0),x.size(1),-1)
        x = torch.mean(x, dim=2) #GlobalAveragePool (average in each channel)
        # experimental FC layer
#        x = self.dropout3(x)
#        x = x.view(x.size(0),-1)
#        x = self.testFC(x)
        
        x = self.block11_2(x)
        
        # we can include these functions in the loss function to save computations
        # but here we do not
#        x = F.softmax(x,dim=1)
        x = torch.sigmoid(x).view(-1) # for single value output
                
        return x
    
# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_net(m):
    if (type(m) == nn.Linear) or (type(m) == nn.modules.conv.Conv3d):
        nn.init.kaiming_uniform_(m.weight)
        
    if hasattr(m, 'bias'):
        try:
            nn.init.constant_(m.bias,0.0)
        except:
            pass


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
        
        # output results
        currPatch = torch.from_numpy(currPatch[None,:,:,:]).to(dtype=dType)
        currLabel = torch.from_numpy(np.array(currLabel)).to(dtype=dType)
        sample = {'image': currPatch, 'labels': currLabel,
                  'seriesuid': currDf['seriesuid'],
                  'coordX': currDf['coordX'],
                  'coordY': currDf['coordY'],
                  'coordZ': currDf['coordZ'],
                  'class' : currDf['class']} # return these values
        
        return sample
    
#%% load the model to be tested
modelToUse = out_path + 'bestDiscriminator_model.pt'
model = discriminatorNet()
model.load_state_dict(torch.load(modelToUse))
model = model.to(device)
model = model.eval()

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

        inputs, labels = data['image'],data['labels']
        inputs = inputs.to(device)
        labels = labels.to(device)
                
        outputs = model(inputs).cpu().numpy()
        
        tmpDf = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','probability'])
        
        for ii in tmpDf.columns[0:-1]:
            tmpDf[ii] = data[ii]
                    
        tmpDf['probability'] = outputs
        
        results_df = results_df.append(tmpDf,ignore_index=False,sort=False).reset_index(drop=True)

results_df.to_csv(results_filename,index=False)