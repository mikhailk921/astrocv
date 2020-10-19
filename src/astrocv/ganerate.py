#!/usr/bin/python
# -*- coding: utf-8 -*-
## @file processing.py
# @brief Файл содержит основные функции генерации графики

import astrocv.castrocv as acv
from astrocv.zernike import *

## @brief - Функция добавления шума
# @param img - исходное изображение
# @param background - средний фон
# @param dNoise - дельта
# @param bpp - bits per pixel
# @param Stride - stride изображения
# @return - Вернет изображение
def addNoiseUniform(img, background, dNoise):
    bpp = int(img.nbytes / img.size) * 8
    if bpp != 8 and bpp != 16 and bpp != 32:
        raise Exception("Unsupported pixel type {}".format(img.dtype))
    Stride = len(img[0]) * (bpp // 8)
    w = len(img[0])
    h = len(img)
    params = [img.data, w, Stride, h, int(background), int(-dNoise), int(+dNoise), bpp]
    w, h = acv.addNoiseUniform(*params)
    return img

## @brief - Функция добавления шума с нормальным распределением
# @param img - исходное изображение
# @param M - математическое ожидание
# @param S - сигма
# @param bpp - bits per pixel
# @param Stride - stride изображения
# @return - Вернет изображение
def addNoiseNorm(img, M, S):
    bpp = int(img.nbytes / img.size) * 8
    if bpp != 8 and bpp != 16 and bpp != 32:
        raise Exception("Unsupported pixel type {}".format(img.dtype))
    Stride = len(img[0]) * (bpp // 8)
    w = len(img[0])
    h = len(img)
    params = [img.data, w, Stride, h, M, S, bpp]
    w, h = acv.addNoiseNorm(*params)
    return img

## @brief - Функция рисования закрашенного круга с распределением яркости по косинусу
# @param img - исходное изображение
# @param X - координата X
# @param Y - координата Y
# @param Diameter - диаметр круга
# @param bpp - bits per pixel
# @param Stride - stride изображения
# @param Signal - яркость круга
# @return - Вернет изображение
def drawObject(img, X, Y, Diameter, Signal=None):
    bpp = int(img.nbytes / img.size) * 8
    if bpp != 8 and bpp != 16 and bpp != 32:
        raise Exception("Unsupported pixel type {}".format(img.dtype))
    Stride = len(img[0]) * (bpp // 8)
    if Signal is None:
        Signal = (1 << bpp) - 1
        if bpp == 32:
            Signal //= 2
    w = len(img[0])
    h = len(img)
    params = [img.data, w, Stride, h, X, Y, Signal, int(Diameter/2), bpp]
    return acv.drawObject(*params)

## @brief - Функция рисования закрашенного круга с распределением яркости по косинусу
# @param img - исходное изображение
# @param X - координата X
# @param Y - координата Y
# @param Diameter - диаметр круга
# @param bpp - bits per pixel
# @param Stride - stride изображения
# @param Signal - яркость круга
# @return - Вернет изображение
def drawObjectPolinomials(img, X, Y, coefficients, power=1e-8, diamAper=0.2, blindZone=0.1,
                          wavelength=0.5e-6, pixSize=5e-6, focus=2.0,exposition=1.0, electronWell=8e3, quantumEffect=1.0):
    bpp = int(img.nbytes / img.size) * 8
    if bpp != 8 and bpp != 16 and bpp != 32:
        raise Exception("Unsupported pixel type {}".format(img.dtype))
    w = len(img[0])
    h = len(img)
    offsetX, offsetY = coefficients[1], coefficients[2]
    coefficients[1] = coefficients[2] = 0
    size = 200
    ampl, poli = makeNearZone(size, coefficients, power, blindZone)
    far = makeFarZone(ampl, poli, diamAper, wavelength, pixSize, focus, exposition, electronWell, quantumEffect)

    left = int(X-len(far)/2)
    right = int(X+len(far)/2)
    top = int(Y - len(far)/2)
    bottom = int(Y + len(far) / 2)
    if left < 0:
        left = 0
    if right > w:
        right = w
    if top < 0:
        top = 0
    if bottom > h:
        bottom = h

    for i in range(0, len(far)):
        img[top + i][left:right] += far[i][0:len(far)]

    return 0

## @brief - Функция выделения объекта на изображении
# @param img - исходное изображение
# @param X - координата X
# @param Y - координата Y
# @param Left - координата левой границы объекта
# @param Top - координата верхней границы объекта
# @param Right - координата правой границы объекта
# @param Bottom - координата нижней границы объекта
# @return - Вернет изображение
def addMarker(img, X, Y, Left, Top, Right, Bottom):
    bpp = int(img.nbytes / img.size) * 8
    if bpp != 8 and bpp != 16 and bpp != 32:
        raise Exception("Unsupported pixel type {}".format(img.dtype))
    Stride = len(img[0]) * (bpp // 8)
    w = len(img[0])
    h = len(img)
    params = [img.data, w, Stride, h, int(X), int(Y), Left, Top, Right, Bottom, bpp]
    w, h = acv.addMarker(*params)
    return img


if __name__ == '__main__':
    import cv2
    import numpy as np

    width = 800
    height = 600

    Noise = 30
    dNoise = 7
    bpp = 8
    type = {8: np.uint8, 16: np.uint16, 32: np.int32}[bpp]
    white_level = (1 << 8) - 1
    if bpp != 8:
        white_level = (1 << 16) - 1

    while True:
        img = np.zeros((height, width), dtype=type)
        #addNoiseNorm(img, Noise, dNoise)
        K = [np.random.random() for i in range(15)]
        #K = [1.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]

        drawObjectPolinomials(img, int(width/2), int(height/2), K, exposition=2)

        cv2.imshow('img', img)

        val = cv2.waitKey(1)
        if val & 0xff == ord('q'):
            exit(0)
