#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/06/03
author: lujie
"""

import argparse
from IPython import embed

root_path = '/Users/relu/data/deep_learning/cs231n'

#----------------------------------------------------------
#   model     |               mode-info                   #
#----------------------------------------------------------
#    vgg      | [A:VGG-11, B:VGG-13, D:VGG-16, E:VGG-19]  #
#----------------------------------------------------------
#   densenet  | [A:DN-121, B:DN-169, C:DN-201, D:DN-264]  #
#----------------------------------------------------------
#  squeezenet | [A:Fire-3-4-1, B:Fire-2-4-2]              #
#----------------------------------------------------------


def config_setting():

    parser = argparse.ArgumentParser('Config for basenet')

<<<<<<< HEAD
    parser.add_argument('--benchmark',  type=str,  default='cifar10')    # default = cifar10
    # ['lenet', 'alexnet', 'vgg', 'inception', 'resnet', 'densenet', 'squeezenet', 'mobilenet']
    parser.add_argument('--base_net',   type=str,  default='mobilenet')

=======
    parser.add_argument('--base_net',   type=str,  default='inception', \
                        choices=['lenet', 'alexnet', 'vgg', 'inception', 'resnet', 'densenet', 'squeezenet'])
    parser.add_argument('--mode',       type=str,  default='A')
>>>>>>> refs/remotes/origin/master
    parser.add_argument('--num_class',  type=int,  default=10)

    parser.add_argument('--platform',   type=str,  default='gpu', choices=['cpu', 'gpu'])   # TODO
    parser.add_argument('--gpus',       type=list, default=[0, 1, 2])                       # TODO
    parser.add_argument('--workers',    type=int,  default=16)                              # TODO

    parser.add_argument('--num_epochs', type=int, default=20)
<<<<<<< HEAD
    parser.add_argument('--batch_size', type=int, default=32)
=======
    parser.add_argument('--batch_size', type=int, default=256)
>>>>>>> refs/remotes/origin/master

    parser.add_argument('--adlr_style',   type=str,   default='all', choices=['all', 'parts'])
    parser.add_argument('--num_saturate', type=int,   default=5)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--optim_md',     type=str,   choices=['sgd','adam','rms'], default='sgd')
    parser.add_argument('--gamma',        type=float, default=0.5)

    parser.add_argument('--save_flag',  type=bool, default=False)
    parser.add_argument('--save_to',    type=str,  default=root_path)

    args = parser.parse_args()

    return args
