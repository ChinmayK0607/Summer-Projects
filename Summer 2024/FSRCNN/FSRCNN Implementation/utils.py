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
from dataset import SRdatasets

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input representation conversion
# source for rgb to ycbcr: https://stackoverflow.com/questions/35595215/conversion-formula-from-rgb-to-ycbcr
# source for ycbcr to rbg: https://en.wikipedia.org/wiki/YCbCr
def convert_rgb_to_y(img, dim_order='chw'):
    
    # ensure that input is batched
    while len(img.shape) < 4:
        img = img.unsqueeze(0) 
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        # dont' care about this case rn
    else:
        ret_tensor = 16. + (64.738 * img[:, 0, :, :] + 129.057 * img[:, 1, :, :] + 25.064 * img[:, 2, :, :]) / 256. 
        # shape = (batch_size, height, width)
        ret_tensor = ret_tensor.unsqueeze(1)
        # shape = (batch_size, 1, height, width)
        return ret_tensor
        # returns 4D tensor

def convert_rgb_to_ycbcr(img, dim_order='chw'):

    # ensure that input is batched
    while len(img.shape) < 4:
        img = img.unsqueeze(0) 

    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[:, 0, :, :] + 129.057 * img[:, 1, :, :] + 25.064 * img[:, 2, :, :]) / 256.
        cb = 128. + (-37.945 * img[:, 0, :, :] - 74.494 * img[:, 1, :, :] + 112.439 * img[:, 2, :, :]) / 256.
        cr = 128. + (112.439 * img[:, 0, :, :] - 94.154 * img[:, 1, :, :] - 18.285 * img[:, 2, :, :]) / 256.
        # shape of each = (batch_size, height, width)
    
    return torch.stack([y, cb, cr], dim = 1) # stacks the tensors along the new 1'th dimension so that the final shape becomes (batch_size, 3, h, w)
    
    # returns 4D tensor

def convert_ycbcr_to_rgb(img, dim_order='chw'):
    
    # ensure that input is batched
    while len(img.shape) < 4:
        img = img.unsqueeze(0) 
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[:, 0, :, :] / 256. + 408.583 * img[:, 2, :, :] / 256. - 222.921
        g = 298.082 * img[:, 0, :, :] / 256. - 100.291 * img[:, 1, :, :] / 256. - 208.120 * img[:, 2, :, :] / 256. + 135.576
        b = 298.082 * img[:, 0, :, :] / 256. + 516.412 * img[:, 1, :, :] / 256. - 276.836
        # shape of each = (batch_size, height, width)
    x = torch.stack([r, g, b], dim = 1) # joins (stacks) along new dimension at position 1
    # shape: (batch_size, 3, height, width)
 
    # min max normalise
    # obtaining min and max values separated for each image and its channels
    # reducing the dimensions -1 and -2 (width and height) in a casacding manner
    # final size of max and min_values: (batch_size, 3, 1, 1) which will be broadcasted to be operated with x
    min_values, _ = torch.min(x, dim = -1, keepdim = True)
    min_values, _ = torch.min(min_values, dim = -2, keepdim = True)
    # expected shape: (batch_size, num_channels, 1, 1)
    max_values, _ = torch.max(x, dim = -1, keepdim = True)
    max_values, _ = torch.max(max_values, dim = -2, keepdim = True)

    epsilon = 1e-17
    # broadcasting expected along dimensions -1 and -2
    x = (x-min_values)/(max_values-min_values + epsilon)
    x = x * 255.0    
    return x
    # returns 4d tensor

import torch
import torch.nn.functional as F

def combine_y_with_cbcr(y_pred, ycbcr_input):
    ycbcr_input.to(device)
    # Extract Y, Cb, and Cr channels
    y = y_pred
    y.to(device) # Shape: (batch_size, 1, height, width)
    
    cb = ycbcr_input[:, 1, :, :]  # Extract Cb channel, shape: (batch_size, height_small, width_small)
    cr = ycbcr_input[:, 2, :, :]  # Extract Cr channel, shape: (batch_size, height_small, width_small)
    
    # Get the height and width of y_pred
    target_height = y_pred.shape[2]  # height of y_pred
    target_width = y_pred.shape[3]   # width of y_pred
    
    # Resize Cb and Cr to the same height and width as y_pred using bicubic interpolation
    # note that the torch.nn.functional.interpolate() expects batched input (batch_size, channel, height, width)
    # hence the unsqueeze inside the function
    cb_resized = F.interpolate(cb.unsqueeze(1), size=(target_height, target_width), mode='bicubic', align_corners=False).to(device)
    cr_resized = F.interpolate(cr.unsqueeze(1), size=(target_height, target_width), mode='bicubic', align_corners=False).to(device)
    # shape: (batch_size, 1, height, width)
    
    # Concatenate the Y, Cb, and Cr channels along the channel dimension (dim=1)
    return torch.cat([y, cb_resized, cr_resized], dim=1)
    # shape: (batch_size, 3, height, width)
    

# defining psnr metric:
class PSNR():
    def single(self, pred, target):

        pred = pred.to(device)
        target = target.to(device)
        while len(pred.shape) < 4:
            pred = pred.unsqueeze(0)
        while len(target.shape) < 4: 
            target = target.unsqueeze(0)

        pred = F.interpolate(pred, size = (480, 320), mode = 'bicubic', align_corners = False)
        
        mse = torch.mean((target - pred) ** 2)
        if mse == 0:
            return 100
        else:
            PIXEL_MAX = 255.0
            mse = mse.item()
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    def batch(self, predictions, target):

        predictions = predictions.to(device)
        target = target.to(device)
        PIXEL_MAX = 255.0

        m = predictions.shape[0]
        mse_acc = 0 # avg mse accumulator
        for i in range(predictions.shape[0]):

            target_inter, predictions_inter = target[i].unsqueeze(0), predictions[i].unsqueeze(0)
            target_inter, predictions_inter = F.interpolate(target_inter, size=(480, 320), mode='bicubic', align_corners=False), F.interpolate(predictions_inter, size=(480, 320), mode='bicubic', align_corners=False)
            # resize required for original LR HR comparison
            mse_acc = mse_acc + torch.mean((target_inter - predictions_inter) ** 2)/m
            
        if mse_acc == 0:
            return 100
        else:
            mse_acc = mse_acc.item()
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_acc))

# prediction (inference) function
def predict(test_loader):
    model.eval()
    with torch.no_grad():
        batch_psnr = 0;
        avg_psnr = 0;
        predictions_list = []
        for i, (low, high) in enumerate(test_loader):
            low = convert_rgb_to_y(low, 'chw')
            low = low.to(device)
            high = convert_rgb_to_y(high, 'chw')
            predictions = model(low)
            
            predictions = predictions.to(device)
            predictions_list.append(predictions.detach().cpu().numpy())
            
            batch_psnr = PSNR().batch(predictions, high)
            avg_psnr = avg_psnr + batch_psnr
            
        avg_psnr = avg_psnr/len(test_loader)
        try:
            predictions = np.array(predictions_list)
            predictions = torch.from_numpy(predictions)
        except:
            # print(predictions_list[0:2])
            predictions = np.array([])
    model.train()
    return avg_psnr, predictions # predictions contains only y_channel

# displaying some randomly picked examples
def display_random(test_loader, model, batch_size = 64):
    print('Some examples: ')
    x = 0 # iterations counter
    for i, (inputs, targets) in enumerate(test_loader):
        for j in np.random.randint(0, batch_size, 10):
            j = min(len(inputs)-1, j) # since last batch might have no. of elements < batch_size if test_size is not a multiple of batch_size
            input_tensor, target_tensor = inputs[j], targets[j]
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)

            # input_arr, target store array with rgb values
            input_arr= np.array(input_tensor)
            target = np.array(target_tensor)
                
            # changing representation from rgb to ycbcr
            ycbcr_input = convert_rgb_to_ycbcr(input_tensor)
            y_input = convert_rgb_to_y(input_tensor)
            y_target = convert_rgb_to_y(target_tensor)

            # passing y_channel to model 
            pred_tensor = model(y_input.to(device))
        
            psnr_input = PSNR().single(pred = y_input, target = y_target)
            psnr_prediction = PSNR().single(pred = pred_tensor, target = y_target)

            # combining y channelled prediction with cbcr of input
            pred_tensor = combine_y_with_cbcr(pred_tensor, ycbcr_input)
            pred_tensor = convert_ycbcr_to_rgb(pred_tensor)

            # pred_arr stores array with rgb values
            pred_arr= pred_tensor.detach().cpu().numpy()
            
            if(len(pred_arr.shape) > 3):
                pred_arr= np.squeeze(pred_arr, 0)
            pred_arr= pred_arr.transpose(1, 2, 0)
    
            # use of prediction list - didn't work out and felt unnecessary:
            '''
            prediction = np.array(predictions[i, j])
            prediction = prediction.transpose(1, 2, 0)
            '''
            if(len(input_arr.shape) > 3):
                input_arr = np.squeeze(input_arr, 0)
            if(len(target.shape) > 3):
                target = np.squeeze(target, 0)
            input_arr = input_arr.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
            
        
            plt.axis('off')
            
            ax = plt.subplot(1, 3, 1)
            ax.imshow(input_arr.astype('uint8'))
            ax.set_title('input')
            plt.text(0, -200, f'input PSNR: {psnr_input:.4f}')
            ax.axis('off')
        
            ax = plt.subplot(1, 3, 2)
            ax.imshow(pred_arr.astype('uint8'))
            ax.set_title('prediction')
            plt.text(0, -100, f'prediction PSNR: {psnr_prediction:.4f}')
            ax.axis('off')
        
            ax = plt.subplot(1, 3, 3)
            ax.imshow(target.astype('uint8'))
            ax.set_title('target')
            ax.axis('off')
            
            plt.show()
            x = x + 1
            if(x == 10):
                break # prints only 10 images
    
# for observing a particular sample (particular image)
def display_particular(i, model):
    sample_set = SRdatasets()
    input_tensor, target_tensor = sample_set[i] # returns unbatched (channels, height, width)
      
    input_tensor_loaded = input_tensor.unsqueeze(0)
    input_tensor_loaded = input_tensor_loaded.to(device)
    ycbcr_input = convert_rgb_to_ycbcr(input_tensor_loaded)
    y_input = ycbcr_input[:, 0, :, :].unsqueeze(1)
    
    prediction = model(y_input.to(device)) # stores y_channel output
    print('input:')
    plt.imshow(np.array(input_tensor).astype('uint8').transpose(1, 2, 0))
    plt.show()
    print('target:')
    plt.imshow(np.array(target_tensor).astype('uint8').transpose(1, 2, 0))
    plt.show()
    print('prediction:')
    prediction = combine_y_with_cbcr(prediction, ycbcr_input)
    prediction = convert_ycbcr_to_rgb(prediction) # outputs batched (4d) tensor
    prediction = prediction.squeeze(0)
    prediction = prediction.detach().cpu().numpy()
    plt.imshow(np.array(prediction).astype('uint8').transpose(1, 2, 0))
    plt.show()
