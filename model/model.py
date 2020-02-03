import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional
import torchvision.datasets
import torchvision.transforms

import numpy as np  # this is torch's wrapper for numpy
from matplotlib import pyplot
from matplotlib.pyplot import subplot
from sklearn.metrics import accuracy_score

"""
using for store the model for training
"""

class Simple_CNN_3d(torch.nn.Module):

    def __init__(self):
        super(Simple_CNN_3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1,bias=True)
        self.bn1 = nn.BatchNorm3d(12)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool_2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(48)
        self.relu3 = nn.ReLU(inplace=True)
        self.max_pool_3 = nn.MaxPool3d(kernel_size=2)

        self.fc1 = nn.Linear(48 * 6 * 6, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        max_pool1 = self.max_pool_1(relu1)

        conv2 = self.conv2(max_pool1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        max_pool2 = self.max_pool_2(relu2)

        conv3 = self.conv3(max_pool2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)
        max_pool3 = self.max_pool_3(relu3)

        x = max_pool3.view(-1, 48 * 6 * 6)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)

        return x




class Simple_CNN(torch.nn.Module):

    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=True)
        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(48 * 6 * 6, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        max_pool1 = self.max_pool_1(relu1)

        conv2 = self.conv2(max_pool1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        max_pool2 = self.max_pool_2(relu2)

        conv3 = self.conv3(max_pool2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)
        max_pool3 = self.max_pool_3(relu3)

        x = max_pool3.view(-1, 48 * 6 * 6)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)

        return x





class Simple_CNN_30(torch.nn.Module):

    def __init__(self):
        super(Simple_CNN_30, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(30)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(50)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=80, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(80)
        self.relu3 = nn.ReLU(inplace=True)
        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(80 * 6 * 6, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        max_pool1 = self.max_pool_1(relu1)

        conv2 = self.conv2(max_pool1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        max_pool2 = self.max_pool_2(relu2)

        conv3 = self.conv3(max_pool2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)
        max_pool3 = self.max_pool_3(relu3)

        x = max_pool3.view(-1, 80 * 6 * 6)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)

        return x




# Defining the network (LeNet-5)
class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(30)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        # Convolution
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(50)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(50 * 12 * 12,2048)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(2048, 1024)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = nn.Linear(1024, 512)  # convert matrix with 84 features to a matrix of 10 features (columns)
        self.fc4 = nn.Linear(512, 2)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        max_pool1 = self.max_pool_1(relu1)

        conv2 = self.conv2(max_pool1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        max_pool2 = self.max_pool_2(relu2)

        x = max_pool2.view(-1, 50 * 12 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)

        return x