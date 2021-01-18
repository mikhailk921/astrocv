#!/astrocv/test/test_objects_detect
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import astrocv as acv
import astrocv.castrocv as cacv
import astrocv.tracker.Tracker as Tracker
from astrocv.processing import getGradientMask, getNoiseMask, integralFrom, getUnionMask
import math as m
import copy


def check_filters(series):
    mask1 = getGradientMask(series)
    mask2 = getNoiseMask(series)
    #mask = getUnionMask(getGradientMask(series), getNoiseMask(series))
    mask = np.minimum(mask1, mask2)
    intmask = integralFrom(mask)

    return mask, intmask

def generate_series_frames(length, w, h, noise, dnoise, bpp):
    series = []
    type = {8: np.uint8, 16: np.uint16, 32: np.int32}[bpp]
    while len(series) < length:
        img = np.zeros((h, w), dtype=type)
        acv.addNoiseNorm(img, noise, dnoise)
        acv.drawObject(img, 400, 400, 10)
        acv.drawObject(img, 450, 450, 15)
        series.append(img)
    return series

width = 800
height = 600

Noise = 30
dNoise = 7
method = acv.search.METHOD_BEST_CONTRAST
minSize = 2
MaxSize = 11
MinCertainty = 10
NMaxObjects = 10
bpp = 8
type = {8: np.uint8, 16: np.uint16, 32: np.int32}[bpp]
white_level = (1 << bpp) - 1

if bpp == 32:
    white_level //= 2

scale = {8: 1, 16: 256, 32: 256}[bpp]
Noise *= scale
dNoise *= scale


mask, intMask = check_filters(generate_series_frames(10, width, height, Noise, dNoise, bpp))

x = y = 300
alpha = 0.0
objects = []
markers = True
while True:
    objects = []
    img1 = np.zeros((height, width), dtype=type)
    acv.addNoiseNorm(img1, Noise, dNoise)

    X = [int(x + 120.0 * m.sin(-alpha)), int(x + 90.0 * m.sin(alpha)),
         int(x + 120.0 * m.sin(alpha / 2.0)), int(x + 150.0 * m.sin(alpha / 3.0))]
    Y = [int(y + 120.0 * m.cos(-alpha)), int(y + 90.0 * m.cos(alpha)),
         int(y + 120.0 * m.cos(alpha / 2.0)), int(y + 150.0 * m.cos(alpha / 1.0))]
    for i in range(0, len(X)):
        acv.drawObject(img1, X[i], Y[i], 10, Signal=50*(i+1)*scale)
    alpha += 0.03

    acv.drawObject(img1, 400, 400, 10)
    acv.drawObject(img1, 450, 450, 15)

    #img1 = acv.smooth(img1, 2)
    #img1 = acv.contrast(img1, 2, 4)


    img1 *= mask
    r = acv.searchObjects(img1, img1, method, minSize, MaxSize, MinCertainty, NMaxObjects, mask=mask, integralMask=intMask)
    #r = acv.searchObjects(img1, img1, method, minSize, MaxSize, MinCertainty, NMaxObjects, mask=None, integralMask=None)

    for i in range(0, len(r)):
        X, Y, Volume, Area = [r[i][n] for n in ["X", "Y", "Volume", "Area"]]
        objects.append(acv.TrackerObject(X, Y, Volume, Area))


    if markers:
        for i in range(0, len(r)):
            acv.addMarker(img1, r[i]["X"], r[i]["Y"],
                          r[i]["Left"], r[i]["Top"], r[i]["Right"], r[i]["Bottom"])

    cv2.imshow("1", img1)
    cv2.imshow("2", mask/mask.max())

    val = cv2.waitKey(20)
    if val & 0xff == ord('q'):
        exit(0)
    elif val & 0xff == 32:  # Space
        markers = not markers
