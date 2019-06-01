#!/usr/bin/env python

# vim: ts=4 sw=4 sts=4 expandtab

import cv2
import numpy as np
import glob
from scipy.ndimage import convolve
from common import config
import os
import shutil
import glob

def bilinear_interpolation(img):
    rb = np.array([[0.25, 0.5, 0.25],
                           [0.5,  0,   0.5],
                           [0.25, 0.5, 0.25]])
    g = np.array([[0, 0.25, 0],
                     [0.25, 0, 0.25],
                     [0, 0.25, 0]])

    img[:, :, 0] = np.where(img[:, :, 0] > 0.0, img[:, :, 0], convolve(img[:, :, 0], rb,mode = 'mirror'))
    img[:, :, 1] = np.where(img[:, :, 1] > 0.0, img[:, :, 1], convolve(img[:, :, 1], g, mode = 'mirror'))
    img[:, :, 2] = np.where(img[:, :, 2] > 0.0, img[:, :, 2], convolve(img[:, :, 2], rb,mode = 'mirror'))

    return img

def generate_rggb(img):
    r_mask = np.zeros(img.shape[:2])
    r_mask[0::2, 0::2] = 1
    b_mask = np.zeros(img.shape[:2])
    b_mask[1::2, 1::2] = 1
    g_mask = np.zeros(img.shape[:2])
    g_mask[0::2, 1::2] = 1
    g_mask[1::2, 0::2] = 1

    img_mosaic = np.zeros(img.shape[:2] +  (3, ))
    img_mosaic[:, :, 0] = b_mask * img[:,:,0]
    img_mosaic[:, :, 1] = g_mask * img[:,:,1]
    img_mosaic[:, :, 2] = r_mask * img[:,:,2]

    return img_mosaic


def prepare_data():
    dataset_path = config.dataset_path
    test_dataset_path = config.test_dataset_path
    train_list = []
    test_list = []
    for single_dataset_path in dataset_path:
        train_list = train_list + glob.glob(single_dataset_path + '/*.png')

    for single_dataset_path in test_dataset_path:
        test_list = test_list + glob.glob(single_dataset_path + '/*.tif')
    total_list = train_list + test_list
    
    for single_image in total_list:
        img = cv2.imread(single_image)
        img = generate_rggb(img)
        img = bilinear_interpolation(img)
        if single_image.endswith('.png'):
            new_name = single_image.split('.png')[0] + '_input.png'
        elif single_image.endswith('.tif'):
            new_name = single_image.split('.tif')[0] + '_input.tif'
        cv2.imwrite(new_name, img)
        


if __name__ == "__main__":
    prepare_data()

