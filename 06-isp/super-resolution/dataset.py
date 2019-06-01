#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from common import config
import glob
from utils import from_img_to_sub_pixel, from_sub_pixel_to_img
from PIL import Image

mat1 = np.array([ 65.481, 128.553, 24.966 ])
mat2 = np.array([-37.797, -74.203, 112.0  ])
mat3 = np.array([ 112.0, -93.786, -18.214])

def rgb2ycbcr(img):
    y = np.round(np.dot(img, mat1) + 16.0)
    cb = np.round(np.dot(img, mat2) + 128.0)
    cr = np.round(np.dot(img, mat3) + 128.0)
    return np.concatenate([y[...,np.newaxis], cb[...,np.newaxis], cr[...,np.newaxis]], axis = 2)

class Dataset():

    def __init__(self, dataset_name):

        test_dataset_path = ['../../dataset/Set14/image_SRF_3']
        train_dataset_path = '../../dataset/Train96'
        self.minibatch_size = config.minibatch_size
        self.ds_name = dataset_name
        self.rng = np.random
        self.ratio = config.ratio
        self.edge = config.edge
        self.stride = config.stride
        self.patch_size = config.patch_size
        train_list = glob.glob(train_dataset_path + '/*.jpg')
        test_list = []
        for single_test_dataset_path in test_dataset_path:
            test_list = test_list + glob.glob(single_test_dataset_path + '/*LR.png')
        self.dataset_meta = {
            'train': (train_list, 4992),
            'test': (test_list, len(test_list)),
        }


    def load(self):

        self.instances = self.dataset_meta[self.ds_name][1]
        self.images = self.dataset_meta[self.ds_name][0]
        self.data_number = len(self.images)
        return self

    @property
    def instances_per_epoch(self):
        return self.instances

    @property
    def minibatchs_per_epoch(self):
        return self.instances // config.minibatch_size

    @property
    def testing_minibatchs_per_epoch(self):
        return self.instances


    def instance_generator(self):
        for i in range(self.instances):
            idx = i % self.data_number
            img = cv2.imread(self.images[idx])
            if self.ds_name == 'test':
                img = rgb2ycbcr(img[:, :, ::-1]/255.0)
                hr_img = rgb2ycbcr(cv2.imread(self.images[idx].split('LR')[0] + 'HR.png')[:, :, ::-1]/255.0)
                yield img.astype(np.float32)/255.0, hr_img.astype(np.float32)/255.0
            elif self.ds_name == 'train':
                h, w = img.shape[:2]
                new_h = h - h % self.ratio
                new_w = w - w % self.ratio
                img = img[0: new_h, 0: new_w, :]
                blurred_img = cv2.GaussianBlur(img, (1,1), 0)
                blurred_img = Image.fromarray(blurred_img.astype(np.uint8))
                lr_img = blurred_img.resize((new_w//config.ratio, new_h//config.ratio),Image.BICUBIC)
                lr_img = np.array(lr_img)
                lr_h, lr_w = lr_img.shape[:2]
                x_start = self.rng.randint(0, (lr_w - self.patch_size)//self.stride + 1 ) * config.stride
                y_start = self.rng.randint(0, (lr_h - self.patch_size)//self.stride + 1 ) * config.stride
                patch = lr_img[y_start:y_start+config.patch_size,x_start:x_start+config.patch_size,:].astype(np.float32)/255.0

                hr_x_start = self.ratio * (x_start + self.edge//2)
                hr_x_end = self.ratio * (x_start + self.patch_size - self.edge//2)
                hr_y_start = self.ratio * (y_start + self.edge//2)
                hr_y_end = self.ratio * (y_start + self.patch_size - self.edge//2)

                hr_img = rgb2ycbcr(img[hr_y_start:hr_y_end, hr_x_start:hr_x_end, ::-1]/255.0)
                hr_patch = from_img_to_sub_pixel(hr_img, self.ratio)
                patch = rgb2ycbcr(patch[:, :, ::-1])

                yield patch.astype(np.float32)/255.0, hr_patch.astype(np.float32)/255.0


if __name__ == "__main__":
    ds = Dataset('train')
    gen = ds.load().instance_generator()

    imggrid = []
    while True:
        for i in range(8):
            img, hr_patch = next(gen)
            img = cv2.resize(img, ((config.patch_size) * config.ratio, (config.patch_size) * config.ratio))
            start_idx = int(config.edge /2 * config.ratio)
            end_idx = int((config.patch_size  - config.edge/2) * config.ratio)
            imggrid.append(img[start_idx:end_idx, start_idx:end_idx, :])
            imggrid.append(from_sub_pixel_to_img(hr_patch, config.ratio))
            
        imggrid = np.array(imggrid).reshape((4, 4, hr_patch.shape[0] * config.ratio, hr_patch.shape[1] * config.ratio, 3))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((4*hr_patch.shape[0] * config.ratio, 4*hr_patch.shape[1] * config.ratio, 3))
        cv2.imshow('', imggrid)
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []

