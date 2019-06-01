#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from common import config
import glob


class Dataset():

    def __init__(self, dataset_name, noise_level=50):

        test_dataset_path = '../../dataset/CBSD68/CBSD68'
        test_noisy_path = '../../dataset/CBSD68/CBSD68_{}'.format(noise_level)
        train_dataset_path = '../../dataset/CBSD432/CBSD432'
        self.minibatch_size = config.minibatch_size
        self.ds_name = dataset_name
        self.rng = np.random
        train_list = glob.glob(train_dataset_path + '/*.jpg')
        test_list = glob.glob(test_dataset_path + '/*.png')
        test_noisy_list = glob.glob(test_noisy_path + '/*.png')
        self.dataset_meta = {
            'train': (train_list, 65536),
            'test': (test_list, len(test_list), test_noisy_list),
        }


    def load(self):

        self.instances = self.dataset_meta[self.ds_name][1]
        self.images = self.dataset_meta[self.ds_name][0]
        if self.ds_name == 'test': self.noisy_images = self.dataset_meta[self.ds_name][2]
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

    def augmentation(self, img):
        s =  config.scale[self.rng.randint(0, 4)]
        h, w = img.shape[:2]
        scale_h = int(h * s)
        scale_w = int(w * s)
        img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_CUBIC)

        return img


    def instance_generator(self):
        for i in range(self.instances):
            idx = i % self.data_number
            img = cv2.imread(self.images[idx])
            if self.ds_name == 'test':
                noisy_img = cv2.imread(self.noisy_images[idx])
                yield img.astype(np.float32)/255.0, noisy_img.astype(np.float32)/255.0
            elif self.ds_name == 'train':
                img = self.augmentation(img)
                h, w = img.shape[:2]
                x_start = 0 + self.rng.randint(0, (w - config.patch_size)//config.stride + 1 ) * config.stride
                y_start = 0 + self.rng.randint(0, (h - config.patch_size)//config.stride + 1 ) * config.stride
                patch = img[y_start:y_start+config.patch_size,x_start:x_start+config.patch_size,:].astype(np.float32)/255.0
                noisy_patch = patch + np.random.normal(0, config.sigma/255.0, patch.shape)
                yield patch.astype(np.float32), noisy_patch.astype(np.float32)


if __name__ == "__main__":
    ds = Dataset('train')
    gen = ds.load().instance_generator()

    imggrid = []
    while True:
        for i in range(8):
            img, noisy_patch = next(gen)
            imggrid.append(img)
            imggrid.append(noisy_patch)
        imggrid = np.array(imggrid).reshape((4, 4, img.shape[0], img.shape[1], img.shape[2]))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((4*img.shape[0], 4*img.shape[1], 3))
        cv2.imshow('', imggrid)
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []

