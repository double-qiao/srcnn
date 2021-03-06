import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import numpy as np

def ssim(img1, img2):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(pre, tar):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    ssim_score = []
    for i in range(pre.shape[0]):
        img1 = pre[i, :, :, :]
        img1 = img1.cpu().numpy()
        img2 = tar[i, :, :, :]
        img2 = img2.cpu().numpy()
        if not img1.shape == img2.shape:
            print(img1.shape)
            print(img2.shape)
            img2 = img2.transpose(0, 2, 1)
            if not img1.shape == img2.shape:
                raise ValueError('Input images must have the same dimensions.')

        if img1.ndim  == 2:
            ssim_score.append(ssim(img1, img2))
            # return ssim(img1, img2)
        elif img1.ndim  == 3:
            if img1.shape[0] == 3:
                ssims = []
                for j in range(3):
                    ssims.append(ssim(img1[j, :, :], img2[j, :, :]))
                ssim_score.append(np.array(ssims).mean())
                # return np.array(ssims).mean()
            elif img1.shape[0] == 1:
                ssim_score.append(ssim(np.squeeze(img1), np.squeeze(img2)))
                # return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    return np.array(ssim_score).mean()

def calculate_psnr(pre, tar):
    # img1 and img2 have range [0, 255]
    psnr_score = []
    for i in range(pre.shape[0]):
        img1 = pre[i, :, :, :]
        img1 = img1.cpu().numpy()
        img2 = tar[i, :, :, :]
        img2 = img2.cpu().numpy()
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        if not img1.shape == img2.shape:
            img2 = img2.transpose(0, 2, 1)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        # return 20 * math.log10(255.0 / math.sqrt(mse))
        psnr_score.append(20 * math.log10(1.0 / math.sqrt(mse)))

    return np.array(psnr_score).mean()
