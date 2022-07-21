
import sys
sys.setrecursionlimit(100000)

# read and write
import os
import sys
import glob
import re
import fnmatch
import csv
import shutil
from datetime import datetime
# maths
import numpy as np
# import pandas as pd
import math
import random
import scipy.ndimage as ndi
import cv2 as cv
from PIL import Image

from skimage.filters import threshold_otsu, threshold_local


# keras
from keras_preprocessing.image import ImageDataGenerator, Iterator
from keras_preprocessing.image.utils import load_img, img_to_array, array_to_img


### UNSUPERVISED MODEL

def model_label(img, p_smooth, posx, posy):
        
    """
    Return the mask of the heart
    """    
    
    im = img_to_array(img)

    # Smooth intensity values
    im_filt = ndi.median_filter(im,  size = p_smooth)

    # Select high-intensity pixels
    mask_start = np.where((im_filt>threshold_otsu(im_filt)) , 1, 0)  
    mask = ndi.binary_closing(mask_start)
    
    # Label the image "mask"
    labels, nlabels = ndi.label(mask)

    # Select left ventricle pixels
    lv_val = labels[posx, posy]
    lv_mask = np.where(labels == lv_val, 1, np.nan)
    
    return labels, lv_mask


def model_seg_heart(img_array, p_smooth, posx, posy):
    
    """
    Take the mask of the heart and replace it by the true pixels of the heart
    """

    labels, lv_mask = model_label(array_to_img(img_array), p_smooth, posx, posy)
    new_img_array = np.ones(img_array.shape)
    for x in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):
            if np.isnan(lv_mask[x,y][1]):
                new_img_array[x,y] = [0,0,0]
            else:
                new_img_array[x,y] = img_array[x,y]
    
    return new_img_array


### FIRST SEGMENTATION

def first_seg(img, label_x, label_y):
    
    """
    Return the segmented image by applying model_seg_heart 2 times
    """

    image = model_seg_heart(model_seg_heart(img_to_array(img), 7, label_x, label_y), 7, label_x, label_y)
    return image


### SECOND SEGMENTATION

def cut(img_arrr):

    """
    Take the image and isolate the heart
    """

    img_arr = img_arrr.copy()
    for x in range(200):
        for y in range(200):
            if x < 50 or x > 170:
                img_arr[x,y] = [0.,0.,0.]
            if y > 150 or y < 30:
                img_arr[x,y] = [0.,0.,0.]
    return img_arr


#### In the first segmentation, some regions outside the heart could be included. The function border1 and border2
#### excludes the parts that are 10 pixels away from the heart.
def border1(bd1, xf, xl, img_arrr):

    """
    Put in black all the region outside the segmented heart for the x-axis
    """

    img_arr = img_arrr.copy()
    for x in range(xf,xl):
        n = 0
        for i in range(1,9):
            if bd1[x]>=0 and img_arr[x,bd1[x]+i][0] == 0 and img_arr[x,bd1[x]+i][1] == 0 and img_arr[x,bd1[x]+i][2]  == 0 :
                n += 1
        if n == 8:  
            for y in range(0,bd1[x]):
                img_arr[x,y]=[0.,0.,0.]
        elif bd1[x] >=0:
            bd1[x] = bd1[x]-1
            border1(bd1, xf, xl, img_arr) 
    return img_arr


def border2(bd2, yf, yl, img_arrr):

    """
    Put in black the region outside the segmented heart for the y-axis 
    """

    img_arr = img_arrr.copy()
    for y in range(yf,yl):
        n = 0
        for i in range(1,9):
            if np.array_equal(img_arr[bd2[y]-i,y],np.array([0.,0.,0.])):
                n += 1
        if n == 8:  
            for x in range(bd2[y],200):
                img_arr[x,y]=[0.,0.,0.]
        elif bd2[y]<199:
            bd2[y] = bd2[y]+1
            border2(bd2, yf, yl, img_arr) 
    return img_arr


def second_seg(img, bd1, bd2, xf, xl, yf, yl, label_x, label_y):

    """
    Return the image segmented after first segmentation and applying cut, border1, border2
    """
    
    image = first_seg(img, label_x, label_y)
    bd1 = [np.nan for i in range(xf)] + [60 for i in range(xf,xl)] + [np.nan for i in range(xl,200)]
    bd2 = [np.nan for i in range(yf)] + [140 for i in range(yf,yl)] + [np.nan for i in range(yl,200)]
    im = cut(image)
    im1 = border1(bd1, xf, xl, im)
    im2 = border2(bd2, yf, yl, im1)
    return image, im2


### VERIFICATION STEPS 

def positions_seg(im_seg_arr):

    """
    Return the positions of the heart's borders
    """

    fpx = [0]*200
    lpx = [199]*200
    fpy = [0]*200
    lpy = [199]*200
    for x in range(200):
        while np.array_equal(im_seg_arr[x,fpx[x]],[0.,0.,0.]):
            fpx[x] += 1
            if fpx[x] == 200:
                break
        while np.array_equal(im_seg_arr[x,lpx[x]],[0.,0.,0.]):
            lpx[x] -= 1
            if lpx[x] == 0:
                break

    for y in range(200):
        while np.array_equal(im_seg_arr[fpy[y],y],[0.,0.,0.]):
            fpy[y] += 1
            if fpy[y] == 200:
                break
        while np.array_equal(im_seg_arr[lpy[y],y],[0.,0.,0.]):
            lpy[y] -= 1
            if lpy[y] == 0:
                break
    return fpx, lpx, fpy, lpy


def cover(img_seg_arr, img):

    """
    Include the surrounding pixels of the heart in the segmented image not to lose information
    """

    im_seg = img_seg_arr.copy()
    im = img_to_array(img)
    fpx, lpx, fpy, lpy = positions_seg(im_seg)
    for x in range(200):
        if fpx[x] != 200 and lpx[x] != 0:
            for y in range(fpx[x]-6, lpx[x]+6):
                if np.array_equal(im_seg[x,y],[0.,0.,0.]):
                    im_seg[x,y] = im[x,y]

    for y in range(200):
        if fpy[y] != 200 and lpy[y] != 0:
            for x in range(fpy[y]-6, lpy[y]+6):
                if np.array_equal(im_seg[x,y],[0.,0.,0.]):
                    im_seg[x,y] = im[x,y]
    return im_seg


def count_pxls_seg(img_seg):
 
    """
    Count the number of pixels highlited by the heart segmented
    """

    n_pxls_seg = 0
    for x in range(200):
        for y in range(200):
            if not(np.array_equal(img_seg[x,y],[0.,0.,0.])):
                n_pxls_seg += 1

    return n_pxls_seg


def metric(img, x_label, y_label):

    """
    Move the position of the pixel used to select the heart mask if the number of pixels of the image segmented is not within the interval considered
    """

    xf, xl = 50, 170
    yf, yl = 30, 150
    bd1 = [np.nan for i in range(xf)] + [60 for i in range(xf,xl)] + [np.nan for i in range(xl,200)]
    bd2 = [np.nan for i in range(yf)] + [140 for i in range(yf,yl)] + [np.nan for i in range(yl,200)]
    image, im2 = second_seg(img, bd1, bd2, xf, xl, yf, yl, x_label, y_label)
    final = cover(im2, img)
    im_final = array_to_img(final)
    n_pxls = count_pxls_seg(final)
 
    if (n_pxls >= 12000 or n_pxls <1500) and (y_label < 125):
        y_label += 5
        im_final, y_label = metric(img, x_label, y_label)
    
    return im_final, y_label





