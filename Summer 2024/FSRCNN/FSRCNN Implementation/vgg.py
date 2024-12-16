import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
import random

import numpy as np
import cv2
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from torchvision import models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg  # Pretrained VGG model
        self.criterion = nn.MSELoss()  # You can also use L2 loss (MSE)
        # Layer weights can emphasize certain layers more (optional)
        self.layer_weights = {
            'relu1_2': 1.0,
            'relu2_2': 1.0,
            'relu3_3': 1.0,
            'relu4_3': 1.0
        }

    def forward(self, generated, target):
        # Extract VGG features for both images
        generated_features = self.vgg(generated)
        target_features = self.vgg(target)
        
        loss = 0
        # Compute perceptual loss across selected layers
        for layer in self.layer_weights:
            loss += self.layer_weights[layer] * self.criterion(
                getattr(generated_features, layer), # generated_features is an instance of VggOutputs class
                getattr(target_features, layer)
            )
        '''
        getattr(object, attr_name) is a Python built-in function
        that returns the attribute value of an object, 
        where the attribute's name is given as a string.
        since attr_name is a variable not an attribute, you cannot directly use object.attr_name
        '''
        
        return loss
