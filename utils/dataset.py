#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import os
import torchvision as tv
import torch.utils.data.dataloader as Data

from .cifar import CIFAR10

root_path = '/Users/relu/data/deep_learning/cs231n/benchmark'


class ToyData(object):

    def __init__(self, args_dict = {}):


        self.batch_size = args_dict.get('batch_size', 32)
        self.workers    = args_dict.get('workers', 4)
        self.benchmark  = args_dict.get('benchmark', 'cifar10')
        self.loader     = None


    def _dataloder(self, train_flag = True, shuffle_flag = False):

        if self.benchmark == 'cifar10':

            data_dir = os.path.join(root_path, 'CIFAR10_data')
            cifar10  = tv.datasets.CIFAR10(data_dir, train = train_flag, \
                              transform = tv.transforms.Compose([
                                  tv.transforms.ToTensor(),
                                  tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ]), download = False)
            # cifar10  = CIFAR10(data_dir, train = train_flag, \
            #                   transform = tv.transforms.Compose([
            #                       tv.transforms.ToTensor(),
            #                       tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #                   ]))
            self.loader = Data.DataLoader(cifar10, batch_size=self.batch_size, \
                              shuffle=shuffle_flag, num_workers=self.workers)
        else:
            pass

        return self.loader
