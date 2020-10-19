#!/astrocv/test/test_objects_detect
# -*- coding: utf-8 -*-

import astrocv.processing as proc
import astrocv.ganerate as gen
import astrocv.search as search
import cv2
import numpy as np

width = 800
height = 800
bpp = 8
type = {8: np.uint8, 16: np.uint16, 32: np.int32}[bpp]
maxSignal = 100
Noise = 30
dNoise = 7

scale = {8: 1, 16: 256, 32: 256}[bpp]
maxSignal *= scale
Noise *= scale
dNoise *= scale

ox = oy = 100
w = h = 300
while True:

    img1 = np.zeros((height, width), dtype=type)
    img2 = np.zeros((height, width), dtype=type)
    gen.addNoiseNorm(img1, Noise, dNoise)
    gen.addNoiseNorm(img2, Noise, dNoise)
    gen.drawObject(img1, 200, 200, 100, maxSignal)
    out = proc.smooth(img1, 20, ox, oy, w, h)
    #out = acv.difference(img1, img2, ox, oy, w, h)
    #out =  acv.add(img1, img2, 1, ox, oy, w, h)
    '''calib = acv.calibrationFrom(img1, 50, ox, oy, w, h)
    calib = np.reshape(np.frombuffer(calib, dtype=type),
                      (h, w))

    out = acv.applyCalibration(img1, calib, ox, oy, w, h)'''
    out = np.reshape(np.frombuffer(out, dtype=type),
                      (h, w))
    little = proc.downsample(out, 4)
    big = proc.upsample(out, 2)
    re = proc.resample(img1, 16)
    conv5 = proc.convolve(img1, proc.get_blur_kernel(2), ox, oy, w, h)
    conv7 = proc.equalize(proc.convolve(little, proc.get_locmax_kernel(3)))
    
    energy = search.energyDistribution(out, 100, 100, np.linspace(0, 100, 10), 0, 0, 200, 200)
    #print("Energy distribution: {}".format(energy))
    power = search.powerDistribution(out, 100, 100, np.linspace(0, 100, 10), 0, 0, 200, 200)
    #print("Power distribution: {}".format(power))

    cv2.imshow("src", img1)
    #cv2.imshow("2", calib)
    cv2.imshow("smooth", out)
    cv2.imshow("downsample", little)
    cv2.imshow("upsample", big)
    cv2.imshow("resample", re)
    cv2.imshow("conv5", conv5)
    cv2.imshow("conv7", conv7)

    val = cv2.waitKey(1)
    if val & 0xff == ord('q'):
        exit(0)
