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

# dataset class

class SRdatasets(Dataset):
    def __init__(self, dataset_path = 'C:/Users/athar/MLprojects/dataloader_task/Datasets', transform = None, scale_factor = 2):
        script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(dataset_path)
        input_list = []
        target_list = []
        
        for dset_name in os.listdir():
        #{
            os.chdir(dataset_path + '\\' + dset_name)
            for dir_name in os.listdir():
            #{
                dir_num = int(dir_name[-1])

                # changing directory to export images
                cwd = os.getcwd()
                os.chdir(cwd + '\\' + dir_name)
                
                num_images = len(os.listdir())
                num_images = num_images//2
                for i in range(1, num_images):
                    input_arr, target = self.extract("LR", dir_num, i), self.extract("HR", dir_num, i)

                    # adding horizontal flip deterministic transform
                    hflip_input = TF.hflip(torch.from_numpy(input_arr))
                    hflip_target = TF.hflip(torch.from_numpy(target))

                    input_list.extend([input_arr, hflip_input])
                    target_list.extend([target, hflip_target])
                    
                # returning back to the dset directory
                os.chdir(dataset_path + '\\' + dset_name)
            #}
        #}
        
        # returning back to the current directory 
        # this way the flow of rest of the program is not affected        
        os.chdir(script_directory)
        
        # converting the list of np.arrays into higher dimenstion np.array since list -> tensor conversion is much slower
        input_arr = np.array(input_list)
        target_arr = np.array(target_list)
        
        self.input_data = torch.Tensor(input_arr) # shape: (num_channels = 3, height, width)
        self.target_data = torch.Tensor(target_arr)
        self.size = len(self.input_data)
        self.transform = transform
    
        

        
    def extract(self, res = "LR", dir_num = 2, i = 1, scale_factor = 2):
        leading_zeros = 3 - len(str(i))
        number_str = leading_zeros * "0" + str(i)

        final_str = "img_" + number_str + "_SRF_" + str(dir_num) + "_" + res + ".png" 
        img = Image.open(final_str)
        npimg = np.asarray(img) # npimg.shape(480, 320, 3) or (320, 480, 3)

        
        # resizing all images into same dimensions
        if res == "LR":
            scale = scale_factor
        else:
            scale = 1
        npimg = cv2.resize(npimg, dsize = (320//scale, 480//scale), interpolation = cv2.INTER_CUBIC)
        npimg = np.array(npimg)
        if len(npimg.shape) == 2:
            npimg = cv2.cvtColor(npimg, cv2.COLOR_GRAY2BGR) 
        
        return np.transpose(npimg, axes = (2, 0, 1)) # returning in form (num_channels = 3, rows, col)
        
    def __getitem__(self, index):
        if isinstance(index, int):
            input_data, target_data = self.input_data[index], self.target_data[index]
            return input_data, target_data
            
        if isinstance(index, slice):
            return torch.stack([(self.input_data[i]) for i in range(*index.indices(len(self)))]), torch.stack([(self.target_data[i]) for i in range(*index.indices(len(self)))])
            # return type: tuple of the form: (tensor of inputs, tensor of targets)
            # shape of each tensor: (size_of_slice, num_channels = 3, height, width)
    
    def __len__(self):
        return self.size