#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : dataset.py
# Purpose : generate samples for train and test
# Creation Date : 2019-03-24 10:35
# Last Modified :
# Created By : niuyazhe
# =======================================

import os
import torch
import numpy as np
from PIL import Image
from scipy import io as scio
from torch.utils.data import Dataset
from torchvision import transforms


class SvhnDataset(Dataset):
    def __init__(self, root, train, transform=None, use_extra_data=True):
        self.root = root
        if train:
            self.data = [os.path.join(root, 'train_32x32.mat')]
            if use_extra_data:
                self.data.append(os.path.join(root, 'extra_32x32.mat'))
        else:
            self.data = [os.path.join(root, 'test_32x32.mat')]

        self.datas_list, self.labels_list = [], []
        for f in self.data:
            samples = scio.loadmat(f)
            self.datas_list.append(samples['X'])
            self.labels_list.append(samples['y'])
        self.datas_np = np.concatenate(self.datas_list, axis=3)
        self.datas_np = self.datas_np.transpose(3, 0, 1, 2)
        self.labels_np = np.concatenate(self.labels_list, axis=0)

        if transform is None:
            raise ValueError
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = self.datas_np[index]
        label = self.labels_np[index]
        label[0] = label[0] % 10
        img = self.transform(Image.fromarray(np.uint8(img)))
        return img.float(), torch.from_numpy(label).long()

    def __len__(self):
        return self.datas_np.shape[0]


if __name__ == "__main__":
    root = '~/data/'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = SvhnDataset(root, train=True, transform=transform)
    img, label = dataset[0]
    print(img.shape)
    print(img.mean())
