#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:49:48 2019

@author: se14
"""
# imports
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.sparse
import scipy.spatial
import pandas as pd
import time
get_ipython().run_line_magic('matplotlib', 'qt')


data_dir  = '/media/se14/DATA/LUNA16/'
cand_path = 'LUNA16_data/candidates_V2.csv'
out_path = '/media/se14/DATA_LACIE/LUNA16/candidates/'
annotations_path = 'LUNA16_data/annotations.csv'
if not os.path.exists(out_path): os.makedirs(out_path)

# %% function definition    
def resample_sitk(image,spacing, new_spacing=[1,1,1]):    

    # reorder sizes as sitk expects them
    spacing_sitk = [spacing[1],spacing[2],spacing[0]]
    new_spacing_sitk = [new_spacing[1],new_spacing[2],new_spacing[0]]
    
    # set up the input image as at SITK image
    img = sitk.GetImageFromArray(image)
    img.SetSpacing(spacing_sitk)                
            
    # set up an identity transform to apply
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(np.eye(3,3).ravel())
    affine.SetCenter(img.GetOrigin())
    
    # make the reference image grid, 80x80x80, with new spacing
    refImg = sitk.GetImageFromArray(np.zeros((80,80,80),dtype=image.dtype))
    refImg.SetSpacing(new_spacing_sitk)
    refImg.SetOrigin(img.GetOrigin())
    
    imgNew = sitk.Resample(img, refImg, affine,sitk.sitkLinear,0)
    
    imOut = sitk.GetArrayFromImage(imgNew).copy()
    
    return imOut


#%%

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "rt") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord // spacing
    return [int(i) for i in voxelCoord]

def normalizePatches(npzarray):
    npzarray = npzarray
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

#%%

# load candidates
cands = readCSV(cand_path)
cands_df = pd.read_csv(cand_path)

# load annotations
annotations_df = pd.read_csv(annotations_path)

#%% get subset folder names to go through
subset_dir = [f'subset{str(ii)}' for ii in range(10)]

ctr_master = 0
start = time.time()
timing_file = 'cand_save_progress/timings.txt'

# for each fold, get the series ids
for kk in range(len(subset_dir)):
    print(kk)
    curr_dir = subset_dir[kk]
    if not os.path.exists(out_path + curr_dir): os.makedirs(out_path + curr_dir)
    
    subset_files = os.listdir(data_dir + curr_dir)
    subset_series_ids = np.unique(np.asarray([subset_files[ll][:-4:] for ll in range(len(subset_files))]))
    
    # initialise an output dataframe with the nodule size (nan for non-nodules)
    out_df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','diameter_mm','filename'])
    
    # load each image (cap to first two for testing)
    ctr = 0
    for jj in range(len(subset_series_ids)):
        image_file = data_dir + curr_dir + '/' + subset_series_ids[jj] + '.mhd'
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(image_file)
        
        curr_cands = cands_df.loc[cands_df['seriesuid'] == subset_series_ids[jj]].reset_index(drop=True)
        
        # go through all candidates that are in this image
        
        # sort to make sure we have all the trues (for prototyping only)
        curr_cands = curr_cands.sort_values('class',ascending=False).reset_index(drop=True)
        
        for cc in range(len(curr_cands)):
            curr_cand = curr_cands.iloc[cc]
        
            # set up info for the output dataframe for this fold
            if curr_cand['class'] == 1:
                # if we have a true nodule, we can get its diameter
                
                # first need to find the corresponding column in the annotations csv (assuming its the closest nodule to the current candidate)
                # extract the annotations for the scan id of our current candidate
                annotations_scan_df = annotations_df.loc[annotations_df['seriesuid'] == curr_cand['seriesuid']]
                
                # extract the coords of the cand and the known nods for the scan
                cand_coord = np.asarray([float(curr_cand[3]),float(curr_cand[2]),float(curr_cand[1])])
                nod_coords = np.array(annotations_scan_df[['coordZ','coordY','coordX']])
                
                # get the distance between the cand and all known nodules
                if nod_coords.ndim == 1: # need to add dims if singleton for cdist
                    nod_coords = [nod_coords]
                    
                distMat = np.squeeze(scipy.spatial.distance.cdist([cand_coord],nod_coords))
                curr_nod = np.argmin(distMat)
                curr_diam = annotations_scan_df.iloc[curr_nod]['diameter_mm']
                
#                print(curr_diam)
            elif curr_cand['class'] == 0:
                # if we don't have a nodule, have no diameter, so can put in a nan
#                continue
                curr_diam = np.nan
                
            tmpDf = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','diameter_mm','filename'])
            tmpDf.loc[0] = [curr_cand['seriesuid'],curr_cand['coordX'],curr_cand['coordY'],curr_cand['coordZ'],curr_cand['class'],curr_diam,f'{out_path + curr_dir}/candidate_{ctr:06d}.raw']
            out_df = out_df.append(tmpDf,ignore_index=True)
            
            # extract and save candidates to raw, with isotropic resampling, and a coverage of 80mm in each dim
            worldCoord = np.asarray([float(curr_cand[3]),float(curr_cand[2]),float(curr_cand[1])])
            voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
            voxelWidth = np.ceil(80 / numpySpacing).astype('uint16')
            
            # pad our image before extracting to avoid cases where index goes out of bounds
            pad_amt = np.ceil(voxelWidth/2 + 1).astype('int16')
            numpyImage_pad = np.pad(numpyImage,((pad_amt[0],pad_amt[0]),(pad_amt[1],pad_amt[1]),(pad_amt[2],pad_amt[2])),mode='reflect')
            
            # then of course need to add these values to our indices to move to new position
            voxelCoord_pad = voxelCoord + pad_amt
            
            patch = numpyImage_pad[voxelCoord_pad[0]-voxelWidth[0]//2:voxelCoord_pad[0]+voxelWidth[0]//2,voxelCoord_pad[1]-voxelWidth[1]//2:voxelCoord_pad[1]+voxelWidth[1]//2,voxelCoord_pad[2]-voxelWidth[2]//2:voxelCoord_pad[2]+voxelWidth[2]//2]
            patch_new = resample_sitk(patch,numpySpacing,new_spacing=[1,1,1])

            patch_new.tofile(f'{out_path + curr_dir}/candidate_{ctr:06d}.raw')
            ctr += 1
            
            ctr_master += 1
            
            if ctr_master == 1000:
                save_time = time.time() - start
                try:
                    f = open(timing_file,'ab')
                except:
                    f = open(timing_file,'xb')
                    
                np.savetxt(f,[save_time])
                f.close()                
                start = time.time()
                ctr_master = 0
            
    out_df.to_csv(f'{out_path + curr_dir}/cand_df_{kk}.csv',index=False)

