#!/usr/bin/env python
# coding: utf-8

# script to test the read speed of two competing methods for reading candidates:
# 1) pre-saved images (resampled)
# 2) resample on-the-fly

# imports
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.sparse
get_ipython().run_line_magic('matplotlib', 'inline')


img_path  = '/home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/Tutorial/TUTORIAL_SimpleITK/TUTORIAL/data/1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689.mhd'
cand_path = '/home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/Tutorial/TUTORIAL_SimpleITK/TUTORIAL/data/candidates.csv'

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
#    
#    image.tofile('debug1.dat')
#    imOut.tofile('debug2.dat')
    
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

#%% load image
numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
print(numpyImage.shape)
print(numpyOrigin)
print(numpySpacing)


# load candidates
cands = readCSV(cand_path)
print(cands)

#%% process and save one candidate to file as a raw, with isotropic resampling, and a coverage of 80mm in each dim
cand = cands[2]
worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
voxelWidth = np.ceil(80 / numpySpacing).astype('uint16')
patch = numpyImage[voxelCoord[0]-voxelWidth[0]//2:voxelCoord[0]+voxelWidth[0]//2,voxelCoord[1]-voxelWidth[1]//2:voxelCoord[1]+voxelWidth[1]//2,voxelCoord[2]-voxelWidth[2]//2:voxelCoord[2]+voxelWidth[2]//2]
patch_new = resample_sitk(patch,numpySpacing,new_spacing=[1,1,1])
#patch_new = normalizePatches(patch_new)
print(patch_new.shape)

patch_new.tofile('example_patch.dat')

#%% try saving in a compressed form
sparse_matrix = scipy.sparse.csc_matrix(patch_new.reshape(80*80,80))
scipy.sparse.save_npz(f'example_patch_conpressed.npz', sparse_matrix, compressed=True)



