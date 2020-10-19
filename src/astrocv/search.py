#!/usr/bin/python
# -*- coding: utf-8 -*-
## @file processing.py
# @brief Файл содержит основные функции поиска объектов на изобажении

import time

import numpy as np

import astrocv.castrocv as acv

METHOD_BEST_INTEGRAL = 0
METHOD_BEST_CONTRAST = 1



## @brief - Функция проверки параметров
# @param img - изображение (numpy.array)
# @param ox - отступ по X
# @param oy - отступ по Y
# @param width - широна выходного изображения
# @param height - высота выходного изображения
# @return - тип изображения, bits per pixel, ox, oy, width, height
def __checkParams(img, ox, oy, width=None, height=None):
    type = img.dtype
    bpp = int(img.nbytes / img.size) * 8
    if bpp != 8 and bpp != 16 and bpp != 32:
        raise Exception("Unsupported pixel type {}".format(img.dtype))
    if width is None:
        width = len(img[0]) - ox
    if height is None:
        height = len(img) - oy
    if width > len(img[0]):
        width = len(img[0])
    if height > len(img):
        height = len(img)
    return type, bpp, ox, oy, width, height

## @brief - Функция поиска объектов на изображении
# @param img_proc - исходное обработанное изображение
# @param img_src - исходное изображение
# @param Method - мотод поиска объектов (0 - METHOD_BEST_INTEGRAL; 1 - METHOD_BEST_CONTRAST)
# @param minSize - минимальный размер объекта
# @param MaxSize - максимальный размер объекта
# @param MinCertainty - минимальный уровень уверенности
# @param NMaxObjects - максимальное количество объектов
# @param bpp - bits per pixel
# @param Stride_proc - stride обработанного изображения
# @param Stride_src - stride исходного изображения
# @return - Вернет изображение
def searchObjects(img_proc, img_src, Method, minSize, MaxSize, MinCertainty, NMaxObjects, mask=None,
                  integralMask=None, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img_proc, Ox, Oy, Width, Height)
    Stride_proc = len(img_proc[0]) * (bpp // 8)
    Stride_src = len(img_src[0]) * (bpp // 8)
    if integralMask is None:
        intMask_bytes = bytearray()
    else:
        intMask_bytes = integralMask.data

    if mask is None:
        mask_bytes = bytearray()
    else:
        mask_bytes = integralMask.data

    params = [img_proc.data, Ox, Oy, Width, Stride_proc, Height, img_src.data, Stride_src,
              mask_bytes, intMask_bytes, Method, minSize,
              MaxSize, MinCertainty, NMaxObjects, bpp]
    count, res, w, h = acv.searchObjects(*params)

    result = []
    for rec in res:
        obj = {}
        for field_i, field_name in enumerate([
                "Left", "Top", "Right", "Bottom",
                "Certainty",
                "AvgX", "AvgY", "MaxX", "MaxY",
                "Diameter", "Area", "Volume",
                "AvgSignal", "StdevSignal", "MaxSignal"
                ]):
            obj[field_name] = rec[field_i]
        obj["X"] = obj["AvgX"]
        obj["Y"] = obj["AvgY"]
        result.append(obj)
    
    return result


## @brief - Функция поиска объектов по заданным участкам ROI на изображении
# @param img_proc - список ROI
# @param img_proc - исходное обработанное изображение
# @param img_src - исходное изображение
# @param Method - мотод поиска объектов (0 - METHOD_BEST_INTEGRAL; 1 - METHOD_BEST_CONTRAST)
# @param minSize - минимальный размер объекта
# @param MaxSize - максимальный размер объекта
# @param MinCertainty - минимальный уровень уверенности
# @param bpp - bits per pixel
# @param Stride_proc - stride обработанного изображения
# @param Stride_src - stride исходного изображения
# @return - Вернет изображение
def searchObjectsForMultiROI(CountROI, ROI, img_proc, img_src, Method, minSize, MaxSize, MinCertainty, mask=None, integralMask=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img_proc, 0, 0)
    Stride_proc = len(img_proc[0]) * (bpp // 8)
    Stride_src = len(img_src[0]) * (bpp // 8)
    if integralMask is None:
        intMask_bytes = bytearray()
    else:
        intMask_bytes = integralMask.data

    if mask is None:
        mask_bytes = bytearray()
    else:
        mask_bytes = integralMask.data
    params = [CountROI, ROI, img_proc.data, Width, Stride_proc, Height, img_src.data, Stride_src, mask_bytes, intMask_bytes, Method, minSize,
              MaxSize, MinCertainty, bpp]
    res, w, h = acv.searchObjectsForMultiROI(*params)

    result = []
    for rec in res:
        obj = {}
        for field_i, field_name in enumerate([
                "Left", "Top", "Right", "Bottom",
                "Certainty",
                "AvgX", "AvgY", "MaxX", "MaxY",
                "Diameter", "Area", "Volume",
                "AvgSignal", "StdevSignal", "MaxSignal"
                ]):
            obj[field_name] = rec[field_i]
        obj["X"] = obj["AvgX"]
        obj["Y"] = obj["AvgY"]
        result.append(obj)

    return result

## @brief - Функция вычисления распределения энергии в заданной зоне
# @param img - исходное изображение
# @param x0 - x-координата точки, относительно которой требуется построить распределение
# @param y0 - y-координата точки, относительно которой требуется построить распределение
# @param R - массив радиусов (в пикселях), для которых определяется доля энергии в круге
# @param Ox - отступ ROI по X
# @param Oy - отступ ROI по Y
# @param Width - ширина ROI
# @param Height - высота ROI
# @return - Вернет массив той же длины, что R, со значениями от 0 до 1 - долей энергии в заданных радиусах
def energyDistribution(img, x0, y0, R, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    stride = len(img[0]) * (bpp // 8)
    x0 -= float(Ox)
    y0 -= float(Oy)
    R = np.array(R, dtype=np.float64)
    I = np.zeros(R.size)
    params = [img.data, stride, Ox, Oy, Width, Height, bpp, float(x0), float(y0), R.data, I.data]
    w, h = acv.energyDistribution(*params)
    return I

## @brief - Функция вычисления распределения плотности мозности в заданной зоне
# @param img - исходное изображение
# @param x0 - x-координата точки, относительно которой требуется построить распределение
# @param y0 - y-координата точки, относительно которой требуется построить распределение
# @param R - массив радиусов (в пикселях), для которых определяется уровень мощности в круге
# @param Ox - отступ ROI по X
# @param Oy - отступ ROI по Y
# @param Width - ширина ROI
# @param Height - высота ROI
# @return - Вернет массив той же длины, что R, со значениями от 0 до 1 - уровнем мощности относительно максимума в заданных радиусах
def powerDistribution(img, x0, y0, R, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    stride = len(img[0]) * (bpp // 8)
    x0 -= float(Ox)
    y0 -= float(Oy)
    R = np.array(R, dtype=np.float64)
    I = np.ones(R.size)
    params = [img.data, stride, Ox, Oy, Width, Height, bpp, float(x0), float(y0), R.data, I.data]
    w, h = acv.powerDistribution(*params)
    return I


class KeyPoint(object):
    def __init__(self, pt, size, area, response, props={}):
        self.pt = pt
        self.size = size
        self.area = area
        self.response = response
        self.props = props


## @brief Класс детектора объектов
class FeatureDetector(object):
    ## @brief - конструктор класса
    # @param method - метод поиска объектов
    # @param minSize - минимальный размер объекта
    # @param maxSize - максимальный размер объекта
    # @param MinCertainty - минимальный уровень уверенности
    # @param NMaxObjects - максимальное количество объектов
    def __init__(self, method=METHOD_BEST_CONTRAST, minSize=3, maxSize=20, minCertainty=0.1, nMaxObjects=5):
        self.method = method
        self.minSize = minSize
        self.maxSize = maxSize
        self.nMaxObjects = nMaxObjects
        self.minCertainty = minCertainty

    ## @brief - Функция поиска объектов
    # @param img - исходное обработанное изображение
    # @param img_src - исходное изображение
    # @return - Вернет словаь объектов
    def detect(self, img, img_src=None, mask=None, intMask=None):
        if img_src is None:
            img_src = img
        
        objects = searchObjects(img, img_src, self.method, self.minSize, self.maxSize, self.minCertainty,
                                self.nMaxObjects, mask=mask, integralMask=intMask)
        
        ret = []
        for props in objects:
            obj = KeyPoint(
                (props["X"], props["Y"]),
                props["Diameter"] / 2,
                props["Area"],
                props["Volume"],
                props
            )
            ret.append(obj)
        return ret

    ## @brief - Функция установки параметров с плавоющей точкой класса
    # @param nameParam - имя параметра
    # @param value - значение параметра
    def setDouble(self, nameParam, value):
        if hasattr(self, nameParam):
            setattr(self, nameParam, value)

    ## @brief - Функция установки целочисленных параметров класса
    # @param nameParam - имя параметра
    # @param value - значение параметра
    def setInt(self, nameParam, value):
        if hasattr(self, nameParam):
            setattr(self, nameParam, value)
        return


def FeatureDetector_create(method):
    m = method.upper()
    if m == 'INTEGRAL':
        return FeatureDetector(method=METHOD_BEST_INTEGRAL)
    elif m == 'CONTRAST':
        return FeatureDetector(method=METHOD_BEST_CONTRAST)
    else:
        raise Exception("Unknown method for object detection: {}".format(method))



