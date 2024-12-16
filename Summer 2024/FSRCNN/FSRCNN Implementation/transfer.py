
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

# transfer model
def transfer_model(PATH, dataset, model, optimizer, epoch_loss_list, epoch_acc_list, train_size, test_size, prev_epochs = 0):
    # transfering previous checkpoint
    try:
    #{
        checkpoint = torch.load(PATH, weights_only = True)
        print('checkpoint loaded successfully')
        transfer = int(input("transfer previous model? 1/0: "))
        if(transfer == 1):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_loss_list.extend(checkpoint['epoch_loss_list'])
            epoch_acc_list.extend(checkpoint['epoch_acc_list'])
            prev_epochs = checkpoint['epoch']
            print('model transfered successfully')     
    #}
    except Exception as e:
        print('Exception occured, running without loading checkpoint')
        checkpnt_flag = 0
        print(e)
    

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return train_set, test_set, prev_epochs # returning prev_epochs is important as in python there is no straightforward provision to modify integers inside functions (passing by reference)