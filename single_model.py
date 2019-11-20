#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import numpy as np


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

     def __init__(self):
         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 5),\
#                                    nn.BatchNorm2d(8),\
#                                    nn.ReLU(),\
#                                    nn.MaxPool2d(2, 2))
#         self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 5),\
#                                    nn.BatchNorm2d(16),\
#                                    nn.ReLU(),\
#                                    nn.MaxPool2d(2, 2),\
#                                    nn.Conv2d(16, 4, 1))
#         self.fc = nn.Linear(100, 10)

         self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 5),\
                                    nn.BatchNorm2d(8),\
                                    nn.ReLU(),\
                                    nn.MaxPool2d(2, 2))
         self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 5),\
                                    nn.BatchNorm2d(8),\
                                    nn.ReLU(),\
                                    nn.Conv2d(8, 16, 1),\
                                    nn.BatchNorm2d(16),\
                                    nn.ReLU(),\
                                    nn.MaxPool2d(2, 2),\
                                    nn.Conv2d(16, 4, 1))
         self.fc = nn.Linear(100, 10)


     def forward(self, x):
         x = self.conv1(x)
         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
         x = x.view(-1, 100)
         x = self.fc(x)

         return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
#        print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()


outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
