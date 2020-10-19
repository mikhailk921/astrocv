#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:26:16 2019

@author: sopenov
"""

import random

import numpy as np

import astrocv.processing
from astrocv.processing import \
    upsample, downsample, resample, \
    upsample_cv2, downsample_cv2, resample_cv2, \
    upsample_pil, downsample_pil, resample_pil
    


np.set_printoptions(edgeitems=4)

astrocv.processing.PREFER_CV2_UPSAMPLE = False
astrocv.processing.PREFER_CV2_DOWNSAMPLE = False

im = np.array(
        [[ 60,  80, 60],
         [ 80, 100, 80],
         [ 60,  80, 60]],
         dtype=np.uint8)
scale = 2


print(im)
print(upsample(im, scale))
print(upsample_cv2(im, scale))
print(upsample_pil(im, scale))

print(downsample(upsample(im, scale), scale))
print(downsample_cv2(upsample_cv2(im, scale), scale))
print(downsample_pil(upsample_pil(im, scale), scale))


M, S = 120, 30
W = H = 512
K = 16
rand_im = np.array([[int(random.gauss(M, S)) for x in range(W)] for y in range(H)], dtype=np.uint8)

print("Original statistics: {} +- {}".format(rand_im.mean(), rand_im.std()))
print("Expected statistics: {} +- {}".format(M, float(S)/float(K)))
scaled_acv = resample(rand_im, K)
print("AstroCV resampled:   {} +- {}".format(scaled_acv.mean(), scaled_acv.std()))
scaled_cv2 = resample_cv2(rand_im, K)
print("OpenCV resampled:    {} +- {}".format(scaled_cv2.mean(), scaled_cv2.std()))
scaled_pil = resample_pil(rand_im, K)
print("PIL resampled:       {} +- {}".format(scaled_pil.mean(), scaled_pil.std()))
