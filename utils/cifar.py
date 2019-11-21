#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from PIL import Image
from .vision import VisionDataset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from IPython import embed


class CIFAR10(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data    = []
        self.targets = []
        self.folder  = 'cifar-10-batches-py'
        self.meta    = {'filename': 'batches.meta', 'key': 'label_names'}

        if train:
            self.dataset = ['data_batch_1', 'data_batch_2', 'data_batch_3',\
                            'data_batch_4', 'data_batch_5']
        else:
            self.dataset = ['test_batch']

        self._process_batches()

        self._load_meta()

    def _process_batches(self):

        for file_name in self.dataset:

            file_path = os.path.join(self.root, self.folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def _load_meta(self):

        path = os.path.join(self.root, self.folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = img.resize((224, 224), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
