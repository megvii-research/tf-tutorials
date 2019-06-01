#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
from common import config
import glob
from scipy.ndimage import convolve

class Dataset():

    def __init__(self, dataset_name):

        dataset_path = config.dataset_path
        test_dataset_path = config.test_dataset_path


        self.minibatch_size = config.minibatch_size
        self.ds_name = dataset_name
        self.rng = np.random
        self.stride = config.stride
        self.patch_size = config.patch_size

        train_list = []
        test_list = []
        for single_dataset_path in dataset_path:
            train_list = train_list + glob.glob(single_dataset_path +'/*_input.png')
        for single_dataset_path in test_dataset_path:
            test_list = test_list + glob.glob(single_dataset_path + '/*_input.tif')


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
            if self.images[idx].endswith('png'):
                gt_img = cv2.imread(self.images[idx].split('_input')[0] + '.png')
            else:
                gt_img = cv2.imread(self.images[idx].split('_input')[0] + '.tif')
            if self.ds_name == 'test':
                yield img.astype(np.float32)/255.0, gt_img.astype(np.float32)/255.0

            elif self.ds_name == 'train':
                h, w = img.shape[:2]
                x_start = self.rng.randint(0, w - self.patch_size + 1 )
                y_start = self.rng.randint(0, h - self.patch_size + 1 )
                patch = img[y_start:y_start+config.patch_size,x_start:x_start+config.patch_size,:].astype(np.float32)/255.0
                gt_patch = gt_img[y_start:y_start+config.patch_size, x_start:x_start+config.patch_size, :].astype(np.float32)/255.0

                yield patch.astype(np.float32), gt_patch.astype(np.float32)


if __name__ == "__main__":
    ds = Dataset('train')
    gen = ds.load().instance_generator()
    imggrid = []
    while True:
        for i in range(8):
            img, gt_img = next(gen)
            imggrid.append(img)
            imggrid.append(gt_img)

        imggrid = np.array(imggrid).reshape((4, 4, img.shape[0], img.shape[1], 3))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((4 * img.shape[0], 4* img.shape[1], 3))
        cv2.imshow('', imggrid)
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []

