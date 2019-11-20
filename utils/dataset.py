#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import os
import torchvision as tv
import torch.utils.data.dataloader as Data

<<<<<<< HEAD
root_path = '/Users/relu/data/deep_learning/cs231n/benchmark'
=======
#root_path = '/home/lujie/Documents/deep_learning/cs231n/cs231n_dataset/'
root_path = '/home/gpu3/lujie/cs231n/cs231n_dataset/'

>>>>>>> refs/remotes/origin/master

class ToyData(object):

    def __init__(self, args_dict = {}):


        self.batch_size = args_dict.get('batch_size', 32)
        self.workers    = args_dict.get('workers', 4)
        self.benchmark  = args_dict.get('benchmark', 'cifar10')
        self.loader     = None


    def _dataloder(self, train_flag = True, shuffle_flag = False):

        if self.benchmark == 'cifar10':

            data_dir = os.path.join(root_path, 'CIFAR10_data')
            self.loader = Data.DataLoader(tv.datasets.CIFAR10(data_dir, train = train_flag, \
                              transform = tv.transforms.Compose([
                                  tv.transforms.ToTensor(),
                                  tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ]), download = False),
                              batch_size=self.batch_size, shuffle=shuffle_flag, num_workers=self.workers)
        else:
            pass

        return self.loader
