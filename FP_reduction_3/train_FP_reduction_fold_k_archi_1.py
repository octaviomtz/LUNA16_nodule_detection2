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
torch.backends.cudnn.deterministic = False#True
torch.backends.cudnn.enabled = True#False
torch.manual_seed(0)
np.random.seed(0)
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dType = torch.float32

# get the current fold from the command line input. 
# for fold k, we use the kth subset as the test, and train on the remaining data
try:
    fold_k = int(sys.argv[1])
    print(f'Training fold {fold_k}')
except:
    print('Defaulting the fold to 0')
    fold_k = 0

#%% paths
cand_path = '/media/se14/DATA_LACIE/LUNA16/candidates/'
out_path = f'results_fold_{fold_k}_archi_1/'
if (not os.path.exists(out_path)) & (out_path != ""): 
    os.makedirs(out_path)

train_subset_folders = [f'subset{i}/' for i in [x for x in range(10) if x!=fold_k]]
train_subset_folders = [cand_path + train_subset_folders[i] for i in range(len(train_subset_folders))]

test_subset_folders = [f'subset{i}/' for i in [x for x in range(10) if x==fold_k]]
test_subset_folders = [cand_path + test_subset_folders[i] for i in range(len(test_subset_folders))]

# set the validation subset
val_subset_folders = [train_subset_folders[fold_k-1]]

# and then remove this from the training subsets
train_subset_folders.remove(val_subset_folders[0]) 

#%% network architecture for FP reduction
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
        
        print(x.shape)
        x = self.C1(x)
        print(x.shape)
        x = self.C2(x)
        print(x.shape)
        x = self.C3(x)
        print(x.shape)
        x = self.D1(x)
        print(x.shape)
        print(x.view(x.shape[0],-1).shape)
        x = self.FC1(x.view(x.shape[0],-1))
        print(x.shape)
        x = self.FC2(x)
        print(x.shape)
        x = torch.sigmoid(x)

        return x
    
model_1 = discriminatorNet_archi_1()
model_1 = model_1.to(dtype=dType).to(device)

print(f'{len(getParams(model_1))} parameters')
    
# initialization function, first checks the module type,
# then applies the desired changes to the weights
#def init_net(m):
#    if (type(m) == nn.Linear) or (type(m) == nn.modules.conv.Conv3d):
#        nn.init.kaiming_uniform_(m.weight)
#        
#    if hasattr(m, 'bias'):
#        try:
#            nn.init.constant_(m.bias,0.0)
#        except:
#            pass

    
#model = discriminatorNet()
#model = model.to(dtype=dType)
#model = model.apply(init_net).to(device)
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
    
    def __init__(self,data_folders,augmentFlag,balanceFlag,n=None):
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
                
        # only set augmentation for training, not validation or testing
        if augmentFlag == True:
            self.augmentFlag = True
        else:
            self.augmentFlag = False
            
        # shuffle repeatably
        cand_df = cand_df.sample(frac=1,replace=False,random_state=fold_k)
            
        # pull out n examples only if possible
        try:
            cand_df = cand_df.iloc[0:n]
        except:
            pass
        
        self.cand_df = cand_df
        
    def __len__(self):
        return 1000
#        return len(self.cand_df)
    
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
            
            # crop out a 40 x 40 x 40 shifted region
            # random translation of up to 5 mm in each direction
            transFact = np.round(10*np.random.rand(3)).astype('int16') - 5
            
        elif self.augmentFlag == False:
            transFact = np.array([0,0,0])
            
        currPatch = currPatch[20+transFact[0]:60+transFact[0],
                              20+transFact[1]:60+transFact[1],
                              20+transFact[2]:60+transFact[2]]
        
        currPatch1 = torch.from_numpy(currPatch[10:-10,10:-10,10:-10][None,:,:,:])
#        currPatch2 = torch.from_numpy(currPatch[5:-5,5:-5,5:-5][None,:,:,:])
#        currPatch3 = torch.from_numpy(currPatch[None,:,:,:])
        
        # output results
        currPatch = torch.from_numpy(currPatch[None,:,:,:])
        currLabel = torch.from_numpy(np.array(currLabel)).to(dtype=dType)
        sample = {'image1': currPatch1, 'labels': currLabel, 'candIdx' : idx} # return these values
        
        return sample

#%% set up dataloader
batch_size = 256
trainData = lidcCandidateLoader(train_subset_folders,augmentFlag=True,balanceFlag=True)
train_dataloader = DataLoader(trainData, batch_size = batch_size,shuffle = True,num_workers = 2,pin_memory=True)

valData = lidcCandidateLoader(val_subset_folders,augmentFlag=False,balanceFlag=False)
val_dataloader = DataLoader(valData, batch_size = batch_size,shuffle = False,num_workers = 2,pin_memory=True)


#%% set up training
criterion = torch.nn.BCELoss()
optimizer_1 = optim.Adam(model_1.parameters(),lr = 6e-6)
ctr = 0
num_epochs = 1
epoch_list = np.array(list(range(num_epochs)))

bestValLoss = 1e6
bestValLossNetFileName = f'bestDiscriminator_model.pt'#_BS{batch_size}_samples{len(trainData)}_epochs{num_epochs}_LR{LR}.pt'

allTrainLoss = np.zeros((num_epochs,1))
allValLoss = np.zeros((num_epochs,1))

optimizer_1.zero_grad()


currModelFilename = f'current_model.pt'

#%% alternative learning rate finders
findLR = False
#findLR = True
if findLR == True:
    print('LR finder')

    allLRs = np.logspace(-7,-1,100)
    LRfinderLoss = np.zeros_like(allLRs).astype('float32')
    
    data = next(iter(train_dataloader))
    # get the inputs
    inputs, labels = data['image1'],data['labels']
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    model_tmp2 = discriminatorNet_archi_1()
    model_tmp2 = model_tmp2.to(dtype=dType).to(device)
    
    for ii, lr in enumerate(allLRs):
        optimizer2 = optim.Adam(model_tmp2.parameters(),lr = allLRs[ii])
        
        # forward + backward + optimize (every numAccum iterations)
        outputs = model_tmp2(inputs) # forward pass
        loss = criterion(outputs[:,0], labels) # calculate loss
        print(f'Batch loss = {loss.item()}')
        
        loss.backward() # backprop the loss to each weight to get gradients
        optimizer2.step() # take a step in this direction according to our optimiser
        optimizer2.zero_grad()

        LRfinderLoss[ii] = loss.item()
    
    plt.figure()    
    plt.semilogx(allLRs,LRfinderLoss)
    plt.title('archi-1')
    
#%% main loop
start = time.time()

# try to load our previous state, if possible
# find the epoch we were up to
if os.path.exists(f'{out_path}lastCompletedEpoch.txt'):
    lastEpoch = np.loadtxt(f'{out_path}lastCompletedEpoch.txt').astype('int16').item()
    epoch_list = epoch_list[epoch_list>lastEpoch]
    print('Found previous progress, amended epoch list')

# load the current model, if it exists
modelToUse = out_path + currModelFilename
if os.path.exists(modelToUse):
    model_1 = discriminatorNet_archi_1()
    model_1.load_state_dict(torch.load(modelToUse))
    model_1 = model_1.to(device)
    print('Loaded previous model')

# set the torch random state to what it last was
if os.path.exists(f'{out_path}randomState.txt'):
    random_state = torch.from_numpy(np.loadtxt(f'{out_path}randomState.txt').astype('uint8'))
    torch.set_rng_state(random_state)
    print('Loaded torch random state')

# load the previous training losses
if os.path.exists(out_path + '/allValLoss.txt') and os.path.exists(out_path + '/allTrainLoss.txt'):
    allValLoss = np.loadtxt(out_path + '/allValLoss.txt')
    allTrainLoss = np.loadtxt(out_path + '/allTrainLoss.txt')
    print('Loaded previous loss history')

print(f'model_1.training = {model_1.training}')

#%%               
for epoch in epoch_list:

    print(f'Epoch = {epoch}')
    running_loss = 0.0

    print('Training')
    for i, data in enumerate(train_dataloader, 0):
        print(f'{i} of {len(train_dataloader)}')
        # get the inputs
        inputs, labels = data['image1'],data['labels']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize (every numAccum iterations)
        outputs = model_1(inputs) # forward pass
        loss = criterion(outputs[:,0], labels) # calculate loss
        print(f'Batch loss = {loss.item()}')

        loss.backward() # backprop the loss to each weight to get gradients
        
        optimizer_1.step() # take a step in this direction according to our optimiser
        optimizer_1.zero_grad()
        
        running_loss += loss.item() # item() gives the value in a tensor
    allTrainLoss[epoch] = running_loss/len(train_dataloader)        

        
    print('Validate')
    with torch.no_grad():
        model = model_1.eval()
        valLoss = 0.0
        for i, data in enumerate(val_dataloader,0):
            
            print(f'{i} of {len(val_dataloader)}')
            loss = 0.
            # get the inputs
            inputs, labels, valIdx = data['image1'],data['labels'],data['candIdx']
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # calculate loss
            outputs = model_1(inputs) # forward pass
            loss = criterion(outputs[:,0], labels).cpu().detach().numpy() # calculate loss
            print(f'Validation loss = {loss.item()}')
            
            valLoss += loss
            
        allValLoss[epoch] = valLoss/len(val_dataloader)
        
        np.savetxt(out_path + '/allValLoss.txt',allValLoss)
        np.savetxt(out_path + '/allTrainLoss.txt',allTrainLoss)
        
        if allValLoss[epoch] < bestValLoss:
            print(f'Best seen validation performance ({bestValLoss} -> {allValLoss[epoch]}), saving...')
            torch.save(model_1.state_dict(),out_path + bestValLossNetFileName)
            np.savetxt(out_path + '/bestEpochNum.txt',np.array([epoch]))
            bestValLoss = allValLoss[epoch]
    
    # checkpointing at the end of every epoch
    torch.save(model_1.state_dict(),out_path + currModelFilename)
    np.savetxt(f'{out_path}lastCompletedEpoch.txt',np.asarray([epoch]))
    np.savetxt(f'{out_path}randomState.txt',torch.get_rng_state().numpy())

    model_1 = model_1.train()
            
    print(f'Epoch = {epoch} finished')
print('Finished Training')
end = time.time()
print(f'Training took {end-start} seconds')















