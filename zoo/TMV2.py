#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
from torch import optim


class Head(nn.Module):

    def __init__(self):

        super(Head, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(32)

    def forward(self, x):
        out = F.relu6(self.bn(self.conv(x)))
        return out


class Tail(nn.Module):
    def __init__(self, num_class):
        super(Tail, self).__init__()
        self.conv_1280 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1280   = nn.BatchNorm2d(1280)
        self.conv_end  = nn.Conv2d(1280, num_class, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_end    = nn.BatchNorm2d(num_class)

    def forward(self, x):
        out = F.relu6(self.bn_1280(self.conv_1280(x)))
        out = F.avg_pool2d(out, kernel_size=7)
        #out = F.relu6(self.bn_end(self.conv_end(out))) # 这里不能这么写，因为当某个通道只有1个值时，使用batchnorm会导致输出结果为0
        out = self.conv_end(out)
        return out

class Bottleneck(nn.Module):#MobileNet_2 网络的Bottleneck层
    n=0
    def __init__(self, in_planes, expansion, out_planes, repeat_times, stride ):

        super(Bottleneck, self).__init__()

        inner_channels = in_planes*expansion
        #Bottlencek3个组件之一:'1*1-conv2d'
        self.conv1      = nn.Conv2d(in_planes, inner_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1        = nn.BatchNorm2d(inner_channels)
        #Bottlencek3个组件之二:dwise
        self.conv2_with_stride = nn.Conv2d(inner_channels,inner_channels,kernel_size=3, stride=stride, padding=1, groups=inner_channels, bias=False) #layer==1 stride=s
        self.conv2_no_stride   = nn.Conv2d(inner_channels,inner_channels,kernel_size=3, stride=1,      padding=1, groups=inner_channels, bias=False) #layer>1  stride=1
        #Bottlencek3个组件之三:linear-1*1-conv2d'
        self.conv3 = nn.Conv2d(inner_channels, out_planes, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        #当某个bottleneck重复出现时，'1*1-conv2d'的输入输出的通道数发生变化，不能再使用conv1了
        self.conv_inner = nn.Conv2d(out_planes, expansion*out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        #当某个bottleneck重复出现时，dwise的输入输出的通道数发生变化，不能再使用conv2_with_stride和conv2_no_stride了
        self.conv_inner_with_stride = nn.Conv2d(expansion*out_planes,expansion*out_planes,kernel_size=3, stride=stride, padding=1, groups=out_planes, bias=False) #layer==1 stride=s
        self.conv_inner_no_stride   = nn.Conv2d(expansion*out_planes,expansion*out_planes,kernel_size=3, stride=1,         padding=1, groups=out_planes, bias=False) #layer>1  stride=1
        #当某个bottleneck重复出现时，'linear-1*1-conv2d'的输入输出的通道数发生变化，不能再使用了
        self.conv3_inner = nn.Conv2d(expansion*out_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        #当某个bottleneck重复出现时，batchnorm的通道数也同样发生了变化
        self.bn_inner = nn.BatchNorm2d(expansion*out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.n = repeat_times

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn1(self.conv2_with_stride(out)))
        out = self.conv3(out)
        out = self.bn2(out)
        count = 2
        while(count<=self.n):
            temp = out
            out = F.relu6(self.bn_inner(self.conv_inner(out)))
            out = F.relu6(self.bn_inner(self.conv_inner_no_stride(out)))
            out = self.conv3_inner(out)
            out = self.bn2(out)
            out = out + temp
            count = count + 1
        return out

class MobileNet2(nn.Module):

    param = [[ 32, 1,  16, 1, 1],
             [ 16, 6,  24, 2, 2],
             [ 24, 6,  32, 3, 2],
             [ 32, 6,  64, 4, 2],
             [ 64, 6,  96, 3, 1],
             [ 96, 6, 160, 3, 2],
             [160, 6, 320, 1, 1]]

    def __init__(self, num_class):

        super(MobileNet2,self).__init__()
        self.layers = self._make_layers(num_class=num_class)

    def _make_layers(self,num_class):
        layer = []
        layer.append( Head() )
        for i in range( len(self.param) ):
            layer.append(Bottleneck(self.param[i][0], self.param[i][1], self.param[i][2], self.param[i][3], self.param[i][4]))
        layer.append( Tail(num_class=num_class) )
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.layers(x)
        return out
