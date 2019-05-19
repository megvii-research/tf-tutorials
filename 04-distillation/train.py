#!/usr/bin/env python
# vim: ts=4 sw=4 sts=4 expandtab

import os
import argparse
import tensorflow as tf

from model import Model
from bigmodel import BigModel
from dataset import Dataset
from common import config
from distillation import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    parser.add_argument('-t', '--temperature', type=float, default=15.0)
    args = parser.parse_args()
    teacher_network = BigModel(args)
    teacher_network.start_session()
    teacher_network.load_from_checkpoint()
    train(args, teacher_network)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)

