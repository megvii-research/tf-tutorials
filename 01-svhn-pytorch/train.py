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
from loss import MaxLoss, LpNorm


def train(network_model, train_loader, test_loader, optimizer, scheduler, criterion, regularizer, args):
    network_model.train()
    model_dir = os.path.join(args.log_model_dir, 'L-{}_N-{}_P-{}'.format(args.loss, args.lp_norm, args.pool))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log = open(os.path.join(model_dir, 'log.txt'), 'w')
    print(args, file=log)

    global_cnt = 0
    for epoch in range(args.epoch):
        scheduler.step()
        for idx, data in enumerate(train_loader):
            global_cnt += 1
            img, label = data
            label = label.squeeze(1)
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
                correct_num = torch.eq(torch.max(output, dim=1)[1], label).sum()
                accuracy = correct_num.cpu().float()/label.shape[0]
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[loss_reg: {:.3f}]\t'.format(reg.item()),
                    '[accuracy: {:.3f}]\t'.format(accuracy),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[loss_reg: {:.3f}]\t'.format(reg.item()),
                    '[accuracy: {:.3f}]\t'.format(accuracy),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0]),
                    file=log
                )
        if epoch % args.test_interval == 0:
            loss_sum = 0.
            acc_sum = 0.
            test_batch_num = 0
            total_num = 0
            for idx, data in enumerate(test_loader):
                test_batch_num += 1
                img, label = data
                total_num += img.shape[0]
                label = label.squeeze(1)
                if torch.cuda.is_available():
                    img, label = img.cuda(), label.cuda()
                output = network_model(img)

                loss = criterion(output, label)
                loss_sum += loss.item()
                acc_sum += torch.eq(torch.max(output, dim=1)[1], label).sum().cpu().float()
            print('\n***************validation result*******************')
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}'.format(acc_sum / total_num)
            )
            print('****************************************************\n')
            print('\n***************validation result*******************', file=log)
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}'.format(acc_sum / total_num),
                file=log
            )
            print('****************************************************\n', file=log)

        if epoch % args.snapshot_interval == 0:
            torch.save(network_model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
    log.close()


def test(network_model, test_loader):
    raise NotImplementedError


def main(args):
    network_model = Model(pool=args.pool)
    if torch.cuda.is_available():
        network_model = network_model.cuda()

    if args.loss == 'L2Loss':
        criterion = nn.MSELoss()
    elif 'CrossEntropy' in args.loss:
        criterion = MaxLoss(args.loss[13:])
    else:
        raise ValueError

    if args.lp_norm == 'None':
        regularizer = None
    else:
        regularizer = LpNorm(args.lp_norm, args.lp_norm_factor)

    optimizer = torch.optim.Adam(network_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestone, gamma=0.5)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = SvhnDataset(root=args.root, train=True, transform=transform)
    test_set = SvhnDataset(root=args.root, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    if args.evaluate:
        test(network_model, test_loader)
        return

    train(network_model, train_loader, test_loader, optimizer, scheduler, criterion, regularizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./')
    parser.add_argument('--log_model_dir', default='./train_log')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--loss', default='CrossEntropy-softmax')
    parser.add_argument('--lp_norm', default=2)
    parser.add_argument('--lp_norm_factor', default=1e-5)
    parser.add_argument('--pool', default='normal')
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--lr_milestone', default=[15, 40])
    parser.add_argument('--epoch', default=60)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--show_interval', default=50)
    parser.add_argument('--test_interval', default=2)
    parser.add_argument('--snapshot_interval', default=5)
    args = parser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)
