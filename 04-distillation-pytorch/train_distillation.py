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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SvhnDataset
from model import Model
from bigmodel import BigModel


class CrossEntropyLoss(nn.Module):
    mode_list = ['normal']

    def __init__(self, num_classes=10, mode='normal', temperature=1):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.criterion_hard = nn.CrossEntropyLoss()
        assert(mode in self.mode_list)
        self.mode = mode

    def forward(self, logits, teacher_output, gt_label):
        if self.mode == 'normal':
            return self.criterion_hard(logits, gt_label)


def train(teacher_model, network_model, train_loader, test_loader, optimizer, scheduler, criterion, args):
    teacher_model.eval()
    network_model.train()
    model_dir = os.path.join(args.log_model_dir, 'L-{}_W-{}_O-{}_T-{}'.format(
        args.loss, args.weight_num_bits, args.output_f_num_bits, args.temperature))
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
            with torch.no_grad():
                output_teacher = teacher_model(img)

            loss = criterion(output, output_teacher, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_cnt % args.show_interval == 0:
                correct_num = torch.eq(torch.max(output, dim=1)[1], label).sum()
                accuracy = correct_num.cpu().float()/label.shape[0]
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[accuracy: {:.3f}]\t'.format(accuracy),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
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
                with torch.no_grad():
                    output_teacher = teacher_model(img)

                loss = criterion(output, output_teacher, label)

                loss_sum += loss.item()
                acc_sum += torch.eq(torch.max(output, dim=1)
                                    [1], label).sum().cpu().float()
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
            torch.save(network_model.state_dict(), os.path.join(
                model_dir, 'epoch-{}.pth'.format(epoch+1)))
    torch.save(network_model.state_dict(), os.path.join(model_dir, 'epoch-final{}.pth'.format(args.epoch)))
    log.close()


def test(network_model, test_loader):
    raise NotImplementedError


def main(args):
    is_train = (args.evaluate == True)
    teacher_model = BigModel()
    network_model = Model(w_num_bits=args.weight_num_bits,
                          out_f_num_bits=args.output_f_num_bits, is_train=is_train)

    if args.load_path is None:
        raise ValueError
    else:
        state_dict = torch.load(args.load_path)
        teacher_model.load_state_dict(state_dict, strict=True)

    if torch.cuda.is_available():
        teacher_model = teacher_model.cuda()
        network_model = network_model.cuda()

    if args.loss == 'CrossEntropyNormal':
        criterion = CrossEntropyLoss(mode='normal')
    else:
        raise ValueError

    def lr_func(epoch):
        lr_factor = args.lr_factor_dict
        lr_key = list(lr_factor.keys())
        index = 0
        for i in range(len(lr_key)):
            if epoch < lr_key[i]:
                index = i
                break
        return lr_factor[lr_key[index]]

    optimizer = torch.optim.Adam(
        network_model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

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

    train(teacher_model, network_model, train_loader,
          test_loader, optimizer, scheduler, criterion, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./')
    parser.add_argument('--load_path', default='teacher.pth')
    parser.add_argument('--log_model_dir', default='./train_log')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--loss', default='CrossEntropyNormal')
    parser.add_argument('--weight_num_bits', default=2)
    parser.add_argument('--output_f_num_bits', default=1)
    parser.add_argument('--temperature', default=15.0)
    parser.add_argument('--init_lr', default=0.01)
    parser.add_argument('--lr_factor_dict', default={15: 1, 40: 0.1, 60: 0.05})
    parser.add_argument('--weight_decay', default=1e-10)
    parser.add_argument('--epoch', default=60)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--show_interval', default=100)
    parser.add_argument('--test_interval', default=2)
    parser.add_argument('--snapshot_interval', default=5)
    args = parser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)
