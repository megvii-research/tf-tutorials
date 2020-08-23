#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : train.py
# Purpose : run a session for training
# Creation Date : 2019-04-18
# Last Modified :
# Created By : niuyazhe
# =======================================


import os
import argparse
from functools import reduce
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import OrderedDict
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from vgg import Vgg16BnCifar
from attack import AttackModel


def total_variation_loss(x, size_average=True):
    assert(isinstance(x, torch.Tensor))
    b, c, h, w = x.shape
    x_base = x[..., :h-1, :w-1]
    x_right = x[..., 1:h, :w-1]
    x_down = x[..., :h-1, 1:w]

    loss = ((x_right-x_base)**2).sum() + ((x_down-x_base)**2).sum()
    if size_average:
        loss /= (b*c*h*w)
    return loss


class MinusCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MinusCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        return (-1) * self.criterion(pred, gt)


def train(args):
    log_dir = os.path.join(args.output_dir, "T{}_L{}_E{}".format(
        args.train_mode,
        args.loss_type,
        args.epoch
    ))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = datasets.CIFAR10(
        root=args.root, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)
    # model
    if args.model == 'vgg16_bn':
        model = Vgg16BnCifar(num_classes=args.num_classes)
    else:
        raise ValueError("invalid model:{}".format(args.model))
    model = AttackModel(model, args.mean, args.std)
    model.cuda()
    model.eval()
    if args.load_path is not None:
        state_dict = torch.load(args.load_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[0] != 'model':
                new_k = 'model.'+k
            else:
                new_k = k
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=True)
        print('load pretrained model in path: {}'.format(args.load_path))
        print('load pretrained model in path: {}'.format(
            args.load_path), file=log_file)

    # criterion(loss)
    loss_dict = {'MinusCrossEntropy': MinusCrossEntropyLoss(),
                 'CrossEntropy': nn.CrossEntropyLoss()}
    if args.train_mode == 'perturb':
        if args.loss_type in loss_dict:
            criterion = loss_dict[args.loss_type]
        else:
            raise NotImplementedError
    elif args.train_mode == 'target':
        raise NotImplementedError
    else:
        raise ValueError('invalid train_mode:{}'.format(args.train_mode))

    pre_noise = torch.zeros(
        args.batch_size, 3, args.img_shape[0], args.img_shape[1])
    pre_noise = pre_noise.cuda()
    pre_noise.requires_grad_(True)

    optimizer = Adam([pre_noise], lr=args.lr, weight_decay=args.weight_decay)

    success_rate = 0.
    noise_l2 = 0.
    for idx, data in enumerate(train_loader):
        img, gt_label = data
        img *= 255
        img, gt_label = img.cuda(), gt_label.cuda()
        img.requires_grad_(False)
        if args.train_mode == 'perturb':
            train_label = gt_label
        elif args.train_mode == 'target':
            raise NotImplementedError

        for epoch in range(args.epoch):
            output, _ = model(pre_noise, img)

            loss = criterion(output, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = torch.eq(gt_label, output.argmax(dim=1)).float()
            print('instance:{}/{}, '.format(idx, args.train_instance_number),
                  'epoch:{} ,'.format(epoch),
                  'loss:{}, '.format(loss.item()),
                  'accuracy:{}'.format(accuracy.item()))
            print('instance:{}/{}, '.format(idx, args.train_instance_number),
                  'epoch:{} ,'.format(epoch),
                  'loss:{}, '.format(loss.item()),
                  'accuracy:{}'.format(accuracy.item()), file=log_file)

        _, adv_example = model(pre_noise, img)
        output = model.evaluate(adv_example)
        accuracy = torch.eq(gt_label, output.argmax(dim=1)).float().item()
        success_rate = (idx*success_rate + 1 - accuracy) / (idx + 1)
        noise_l2 = (idx*noise_l2 + torch.norm(img -
                                              adv_example, p=2).item()) / (idx + 1)

        if idx >= args.train_instance_number:
            print('enough train instance, end train procedure...')
            print('enough train instance, end train procedure...', file=log_file)
            break

    print('final success_rate:{}, noise_l2:{}'.format(success_rate, noise_l2))
    print('final success_rate:{}, noise_l2:{}'.format(
        success_rate, noise_l2), file=log_file)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./experiment/')
    parser.add_argument('--model', default='vgg16_bn')
    parser.add_argument('--root', default='./', help='path to cifar-10 dataset')
    parser.add_argument('--load_path', default='epoch_99_cifar10_baseline.pth',
                        help='pretrained model state_dict path')
    parser.add_argument('--img_shape', default=(32, 32))
    parser.add_argument('--mean', default=120.707)
    parser.add_argument('--std', default=64.15)
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--train_instance_number', default=100)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--weight_decay', default=1e-10)
    parser.add_argument('--epoch', default=500)
    parser.add_argument('--train_mode', default='perturb',
                        help='perturb mode or target mode')
    parser.add_argument('--loss_type', default='MinusCrossEntropy')
    parser.add_argument('--save_interval', default=10)
    parser.add_argument('--show_interval', default=1)
    args = parser.parse_args()
    train(args)
