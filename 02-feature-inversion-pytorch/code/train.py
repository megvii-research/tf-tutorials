#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : train.py
# Purpose : run a session for training
# Creation Date : 2019-04-10
# Last Modified :
# Created By : niuyazhe
# =======================================


import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.optim import Adam
from vgg import vgg16_bn
from resnet import resnet18


def load_img(path, img_shape=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, img_shape)
    img = (img/255).astype(np.float32)
    return img


def sigmoid(x):
    return (1./(1+np.exp(-x)))


def output_img(img, name):
    assert(isinstance(img, np.ndarray))
    img = (img*255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(name, img)


def extract_feature(model, inputs, layer_str, transform_dict):
    feature_map = torch.Tensor().cuda()

    def hook(module, input_feature, output_feature):
        feature_map.resize_(output_feature.shape)
        feature_map.copy_(output_feature)

    layer = model
    for item in transform_dict[layer_str]:
        layer = getattr(layer, item)
    handle = layer.register_forward_hook(hook)
    _ = model(inputs)
    handle.remove()
    return feature_map


class FeatureMatchLpLoss(nn.Module):
    def __init__(self, p=2):
        super(FeatureMatchLpLoss, self).__init__()
        self.p = p

    def forward(self, inputs, target):
        return torch.norm(torch.abs(inputs-target), p=self.p)


def train(args):
    log_dir = os.path.join(args.output_dir, "M{}_L{}_W{}_F{}".format(
        args.model,
        str(args.lr)+"-".join([str(x) for x in args.lr_milestones]),
        args.weight_tv,
        "-".join(args.feature)
    ))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    transform_dict = None
    if args.model == 'vgg16_bn':
        model = vgg16_bn(pretrained=True)
        transform_dict = {'conv3_1': ('features', '10')}
    elif args.model == 'resnet18':
        model = resnet18(pretrained=True)
    else:
        raise ValueError("invalid model:{}".format(args.model))
    assert(transform_dict is not None)
    model.cuda()
    model.eval()

    img = load_img(args.img_path, args.img_shape)
    pre_noise = np.random.uniform(
        low=-3, high=3, size=img.shape).astype(np.float32)
    pre_noise = sigmoid(pre_noise)
    img_tensor = torch.from_numpy(img).permute(
        2, 0, 1).contiguous().unsqueeze(0).cuda()
    noise_tensor = torch.from_numpy(pre_noise).permute(
        2, 0, 1).contiguous().unsqueeze(0).cuda()
    noise_tensor.requires_grad_(True)

    criterion = FeatureMatchLpLoss(p=2)

    optimizer = Adam([noise_tensor], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=0.5)

    for epoch in range(args.epoch):
        scheduler.step()
        loss = torch.Tensor().cuda()
        for item in args.feature:
            img_output = extract_feature(
                model, img_tensor, item, transform_dict)
            noise_output = extract_feature(
                model, noise_tensor, item, transform_dict)
            loss += criterion(noise_output, img_output)

        loss = criterion(img_output, noise_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.show_interval == 0:
            print("e:{}---loss:{:.5f}".format(epoch, loss.item()))
            print("e:{}---loss:{:.5f}".format(epoch, loss.item()), file=log_file)
        if epoch % args.save_interval == 0:
            noise_np = noise_tensor.data.cpu().squeeze(
                0).permute(1, 2, 0).contiguous().numpy()
            output_img(noise_np, os.path.join(
                log_dir, "epoch_{}.png".format(epoch)))
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='../experiment/')
    parser.add_argument('--model', default='vgg16_bn')
    parser.add_argument('--img_path', default='../images/face.jpg')
    parser.add_argument('--img_shape', default=(224, 224))
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--lr_milestones', default=[400, 800, 1700])
    parser.add_argument('--weight_tv', default=0)
    parser.add_argument('--feature', default=['conv3_1'])
    parser.add_argument('--epoch', default=2000)
    parser.add_argument('--save_interval', default=10)
    parser.add_argument('--show_interval', default=10)
    args = parser.parse_args()
    train(args)
