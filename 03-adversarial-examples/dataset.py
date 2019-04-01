#!/usr/bin/env python

import os
import cv2
import numpy as np
import pickle
from common import config

class Dataset():
    dataset_path = '../../cifar10-dataset/'


    def __init__(self, dataset_name):
        self.ds_name = dataset_name
        self.minibatch_size = config.minibatch_size
        self.rng = np.random
        self.instances = config.nr_instances
        if self.ds_name == 'train':
            self.files = [self.dataset_path + 'data_batch_{}'.format(i + 1) for i in range(5)]
        else:
            self.files = [self.dataset_path + 'test_batch']
    def load(self):
        datas_list, labels_list = [], []
        for file in self.files:
            with open(file, 'rb') as f:
                samples = pickle.load(f, encoding = 'bytes')
                datas_list.extend(samples[b'data'])
                labels_list.extend(samples[b'labels'])
        self.samples_mat = {'X': datas_list, 'Y': labels_list}
        return self

    @property
    def total_instances(self):
        return self.instances

    @property
    def minibatches(self):
        return self.instances // config.minibatch_size

    def instance_generator(self):
        for i in range(self.instances):
            img_r = self.samples_mat['X'][i][:1024].reshape(config.image_shape[0], config.image_shape[1], 1)
            img_g = self.samples_mat['X'][i][1024:2048].reshape(config.image_shape[0], config.image_shape[1], 1)
            img_b = self.samples_mat['X'][i][2048:].reshape(config.image_shape[0], config.image_shape[1], 1)
            img = np.concatenate((img_r, img_g, img_b), axis = 2)
            label = self.samples_mat['Y'][i]

            yield img.astype(np.float32), np.array(label, dtype=np.int32)


if __name__ == "__main__":
    ds = Dataset('train')
    ds = ds.load()
    gen = ds.instance_generator()

    imggrid = []
    while True:
        for i in range(25):
            img, label = next(gen)
            img = cv2.resize(img, (96, 96))
            cv2.putText(img, str(label), (0, config.image_shape[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            imggrid.append(img)

        imggrid = np.array(imggrid).reshape((5, 5, img.shape[0], img.shape[1], img.shape[2]))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5*img.shape[0], 5*img.shape[1], 3))

        cv2.imshow('', imggrid.astype('uint8'))
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []

