#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : train.py
# Purpose : build a compute graph
# Creation Date : 2019-03-23 23:28
# Last Modified :
# Created By : niuyazhe
# =======================================


import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SvhnDataset
from model import Model
from loss import Softmax, LpNorm


def train(network_model, train_loader, test_loader, optimizer, scheduler, criterion, regularizer, args):
    model_dir = os.path.join(args.log_model_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    global_cnt = 0
    for epoch in range(args.epoch):
        scheduler.step()
        for idx, data in enumerate(train_loader):
            global_cnt += 1
            img, label = data
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            output = network_model(img)

            loss = criterion(output, label)
            if regularizer is not None:
                reg = regularizer(network_model)
                loss += reg
            else:
                reg = torch.Tensor([0.])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_cnt % args.show_interval == 0:
                accuracy = torch.eq(torch.max(output, dim=1), label).sum() / label.shape[0]
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[loss_reg: {:.3f}]\t'.format(reg.item()),
                    '[accuracy: {:.3f}]\t'.format(accuracy.item()),
                    '[lr: {:.3f}]\n'.format(scheduler.get_lr().item())
                )
        if epoch % args.test_interval == 0:
            loss_sum = 0.
            acc_sum = 0.
            test_batch_num = 0
            for idx, data in enumerate(test_loader):
                test_batch_num += 1
                img, label = data
                if torch.cuda.is_available():
                    img, label = img.cuda(), label.cuda()
                output = network_model(img)

                loss = criterion(output, label)
                loss_sum += loss_sum.item()
                acc_sum += torch.eq(torch.max(output, dim=1), label).sum().item()
            print('\n***************validation result*******************')
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}\n'.format(acc_sum / test_batch_num)
            )
            print('****************************************************\n')

        if epoch % args.snapshot_interval == 0:
            torch.save(network_model.state_dict(), os.path.join(model_dir, 'epoch-{}'.format(epoch)))


def test(network_model, test_loader):
    raise NotImplementedError


def main(args):
    network_model = Model()
    if torch.cuda.is_available():
        network_model = network_model.cuda()

    if args.loss == 'softmax':
        criterion = Softmax()
    else:
        raise ValueError

    if args.lp_norm == None:
        regularizer = None
    else:
        raise ValueError

    optimizer = torch.optim.Adam(network_model.parameters(), lr=args.lr)
    scheduler = torch.optim.MultiStepLR(optimizer, args.lr_milestone, gamma=0.1)

    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    train_set = SvhnDataset(root=args.root, train=True, transform=transform)
    test_set = SvhnDataset(root=args.root, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if arg.evaluate:
        test(network_model, test_loader)
        return

    train(network_model, train_loader, test_loader, optimizer, scheduler, criterion, regularizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./')
    parser.add_argument('--log_model_dir', default='./train_log')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--loss', default=None)
    parser.add_argument('--lp_norm', default=None)
    parser.add_argument('--lp_norm_factor', default=1e-5)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--lr_milestone', default=[15, 40])
    parser.add_argument('--epoch', default=60)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--show_interval', default=100)
    parser.add_argument('--test_interval', default=2)
    parser.add_argument('--snapshot_interval', default=1)
    args = parser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    main(args)
