#!/usr/bin/python
# -*- coding: utf-8 -*-
## @file processing.py
# @brief Файл содержит основные функции графической библиотеки
import random

import numpy as np

import astrocv.castrocv as acv

PREFER_CV2_UPSAMPLE = False
PREFER_CV2_DOWNSAMPLE = False
PREFER_CV2_CONVOLVE = False

METHOD_BEST_INTEGRAL = 0
METHOD_BEST_CONTRAST = 1


def max_threads_count():
    return acv.max_threads_count()


def set_threads_count(count):
    return acv.set_threads_count(count)


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


## @brief - Функция вычисления интегрального изображения
# @param src - исходное изображение
# @return - Вернет интегральное изображение
def integralFrom(src):
    integral = np.zeros(np.shape(src), dtype=np.int32)
    bpp = int(src.nbytes / src.size) * 8
    h, w = np.shape(src)
    stride = w * bpp // 8
    params = [src.data, integral.data, w, h, stride, bpp]
    w, h = acv.integralFrom(*params)

    return integral


## @brief - Функция сложения изображений одинокового размера
# @param first_img - первое изображение
# @param second_img - второе изображение
# @param bpp - bits per pixel
# @param Stride_A - stride первого изображения
# @param Stride_B - stride второго изображения
# @param Stride_C - stride выходного изображения
# @return - Вернет изображение
def add(first_img, second_img, koef=1, Ox=0, Oy=0, Width=None, Height=None):
    return __add_sub(first_img, second_img, koef, Ox=Ox, Oy=Oy, Width=Width, Height=Height)

## @brief - Функция вычитания изображений одинокового размера
# @param first_img - первое изображение
# @param second_img - второе изображение
# @param bpp - bits per pixel
# @param Stride_A - stride первого изображения
# @param Stride_B - stride второго изображения
# @param Stride_C - stride выходного изображения
# @return - Вернет изображение
def sub(first_img, second_img, koef=-1, Ox=0, Oy=0, Width=None, Height=None):
    return __add_sub(first_img, second_img, koef, Ox=Ox, Oy=Oy, Width=Width, Height=Height)


def __add_sub(first_img, second_img, koef, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(first_img, Ox, Oy, Width, Height)
    StrideFir = len(first_img[0]) * bpp // 8
    StrideSec = len(second_img[0]) * bpp // 8
    StrideOut = Width * bpp // 8
    out = np.zeros((Height, Width), dtype=type)
    params = [first_img.data, second_img.data, Ox, Oy, Width, Height, StrideFir, StrideSec, out.data, StrideOut, koef, bpp]
    w, h = acv.add(*params)
    return out

## @brief - Функция вычисления разности двух последних изображений
# @param img - первое изображение
# @param last_img - второе изображение
# @param bpp - bits per pixel
# @param Stride - stride изображений
# @return - Вернет изображение
def difference(img, last_img, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    StrideIn = len(img[0]) * bpp // 8
    StrideOut = Width * bpp // 8
    out = np.zeros((StrideOut * Height), dtype=type)
    params = [img.data, last_img.data, Ox, Oy, Width, StrideIn, Height, out.data, StrideOut, bpp]
    w, h = acv.difference(*params)
    return out

## @brief - Функция сглаживания
# @param img - исходное изображение
# @param D - величина сглаживания
# @param bpp - bits per pixel
# @param Stride - stride изображений
# @return - Вернет изображение
def smooth(img, D, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    StrideIn = len(img[0]) * (bpp // 8)
    StrideOut = Width * (bpp // 8)

    out = np.zeros((Height, Width), dtype=type)
    params = [img.data, StrideIn, Ox, Oy, Width, Height, out.data, StrideOut, int(D), bpp]
    w, h = acv.smooth(*params)
    return out

## @brief - Функция пространственного фильтра
# @param img - исходное изображение
# @param Dmin - размер внутренней области
# @param Dmax - размер внешней области
# @param Stride - stride изображений
# @return - Вернет изображение
def contrast(img, Dmin, Dmax, Ox=0, Oy=0, Width=None, Height=None, mask=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    StrideIn = len(img[0]) * bpp // 8
    StrideOut = Width * bpp // 8
    if mask is None:
        mask_bytes = bytearray()
    else:
        mask_bytes = mask.data
    out = np.zeros((Height, Width), dtype=type)
    params = [img.data, StrideIn, Ox, Oy, Width, Height, mask, out.data, StrideOut, int(Dmin), int(Dmax), bpp]
    w, h = acv.contrast(*params)
    return out


## @brief Негативное изображение (инверсия)
# @param img - исходное изображение
# @return - инвертированное изображение (негативное)
def negative(img, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    white_level = (1 << bpp) - 1
    if Ox == 0 and Oy == 0 and Width == len(img[0]) and Height == len(img):
        img = white_level - img
    else:
        img[Ox:Ox+Width, Oy:Oy+Height] = white_level - img[Ox:Ox+Width, Oy:Oy+Height]
    return img

## @brief Эквализация гистрограммы ("растяжка")
#
#  Производит гистограммное преобрахование уровней яркости,
#  сопоставляя самый тусклый пиксель чёрному цвету, а самый
#  яркий пиксель белому цвету.
#
# @param img - исходное изображение
# @return - эквализированное изображение
def equalize(img, Ox=0, Oy=0, Width=None, Height=None):
    roi = None
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    white_level = (1 << bpp) - 1
    if Ox == 0 and Oy == 0 and Width == len(img[0]) and Height == len(img):
        min_l, max_l = img.min(), img.max()
    else:
        roi = img[Ox:Ox+Width, Oy:Oy+Height]
        min_l, max_l = roi.min(), roi.max()
    offset = -int(min_l)
    scale = int(white_level / max(1, max_l - min_l))
    if Ox == 0 and Oy == 0 and Width == len(img[0]) and Height == len(img):
        img = np.array((img + offset) * scale, dtype=type)
    else:
        img[Ox:Ox+Width, Oy:Oy+Height] = np.array((roi + offset) * scale, dtype=type)
    return img

## @brief Вычет цвета (канала)
#
#  Вычитает из одного канала цвета другой канал, при этом
#  используется усеченное вычитание (отрицательные значения
#  приравниваются к нулю).
#
# @param chan1 - первый канал (из которого производится вычитание)
# @param chan2 - второй канал (который вычитается)
# @return - результат вычитания
def sub_channel(chan1, chan2, Ox=0, Oy=0, Width=None, Height=None):
    if Ox == 0 and Oy == 0 and Width == len(chan1[0]) and Height == len(chan1):
        s1, s2 = chan1.sum(), chan2.sum()
    else:
        s1 = chan1[Ox:Ox+Width, Oy:Oy+Height].sum()
        s2 = chan2[Ox:Ox+Width, Oy:Oy+Height].sum()
    koef = -s1 / max(1, s2)
    return add(chan1, chan2, koef, Ox, Oy, Width, Height)


## @brief Обесцвечивание изображения
# @param colors - каналы цветов изображения
# @return - монохромное изображение
def grayscale(colors):
    ret = colors[0]
    for i in range(1, len(colors)):
        ret = ret + colors[i]
    return ret / float(len(colors))


## @brief - Функция вычисления калибровочного кадра
# @param img - исходное изображение
# @param Smoothed - величина сглаживания
# @param Stride_src - stride исходного изображения
# @param Stride_out - stride выходного изображения
# @return - Вернет изображение
def calibrationFrom(img, Smoothed, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    StrideIn = len(img[0]) * bpp // 8
    StrideOut = Width * bpp // 8
    out = np.zeros((StrideOut * Height), dtype=type)
    params = [img.data, StrideIn, Ox, Oy, Width, Height, out.data, StrideOut, Smoothed, bpp]
    w, h = acv.calibrationFrom(*params)
    return out

## @brief - Функция калибровки
# @param img - исходное изображение
# @param img_calib - калибровочное изображение
# @param Stride_src - stride исходного изображения
# @param Stride_calib - stride калибровочного изображения
# @param Stride_out - stride выходного изображения
# @return - Вернет изображение
def applyCalibration(img, imgCalib, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    StrideIn = len(img[0]) * bpp // 8
    StrideCalib = len(imgCalib[0]) * bpp // 8
    StrideOut = Width * bpp // 8
    out = np.zeros((StrideOut * Height), dtype=type)
    params = [img.data, StrideIn, Ox, Oy, Width,  Height, imgCalib.data, out.data, bpp]
    w, h = acv.applyCalibration(*params)
    return out


## @brief - Функция анализа изображения
# @param img - исходное изображение
# @param Ox - отступ по X
# @param Oy - отступ по Y
# @param Width - широна изображения
# @param Height - высота изображения
# @return - информацию о изображении (avg, min, max, stdev)
def imageInfo(img, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(img, Ox, Oy, Width, Height)
    StrideIn = len(img[0]) * bpp // 8
    params = [img.data, StrideIn, Ox, Oy, Width,  Height, bpp]
    avg, min, max, stdev = acv.image_info(*params)
    return avg, min, max, stdev


_GAUSS_KERNELS = {}
_BLUR_KERNELS = {}
_LOCMAX_KERNELS = {}

def normalize_kernel(kernel):
    h, w = kernel.shape
    avg = kernel.sum() / float(h*w)
    return kernel - avg

def get_gauss_kernel(ksize):
    global _GAUSS_KERNELS
    ksize = int(ksize)
    if ksize not in _BLUR_KERNELS:
        kernel = np.zeros((ksize, ksize))
        R = float(ksize) * 0.5
        R2 = R*R
        x0 = y0 = R
        for y in range(ksize):
            for x in range(ksize):
                dx, dy = float(x) + 0.5 - x0, float(y) + 0.5 - y0
                kernel[x, y] = np.exp(-(dx*dx + dy*dy) / R2)
        _GAUSS_KERNELS[ksize] = kernel
    return _GAUSS_KERNELS[ksize]

def get_blur_kernel(ksize):
    global _BLUR_KERNELS
    ksize = int(ksize)
    if ksize not in _BLUR_KERNELS:
        blur_kernel = get_gauss_kernel(2*ksize + 1)
        koef = 1.0 / float(blur_kernel.size)
        _BLUR_KERNELS[ksize] = blur_kernel * koef
    return _BLUR_KERNELS[ksize]

def get_locmax_kernel(ksize):
    global _LOCMAX_KERNELS
    ksize = int(ksize)
    if ksize not in _LOCMAX_KERNELS:
        locmax_kernel = get_gauss_kernel(2*ksize + 1)
        locmax_kernel = normalize_kernel(locmax_kernel)
        _LOCMAX_KERNELS[ksize] = locmax_kernel
    return _LOCMAX_KERNELS[ksize]


try:
    import cv2
    
    _HAS_CV2 = True
    
    def _scale_cv2(src, factor, Ox=0, Oy=0, Width=None, Height=None):
        type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
        if Ox != 0 or Oy != 0 or Height != len(src) or Width != len(src[0]):
            src = src[Oy:Oy+Width, Ox:Ox+Height]
        shape = (int(Height*factor), int(Width*factor))
        dst = cv2.resize(src, shape, interpolation=cv2.INTER_LINEAR)
        return dst
    
    def upsample_cv2(src, factor, Ox=0, Oy=0, Width=None, Height=None):
        return _scale_cv2(src, factor, Ox, Oy, Width, Height)
    
    def downsample_cv2(src, factor, Ox=0, Oy=0, Width=None, Height=None):
        return _scale_cv2(src, 1.0 / factor, Ox, Oy, Width, Height)
    
    def resample_cv2(src, down=4, up=None):
        if up is None:
            up = down
        ret = src
        if down != 1:
            ret = downsample_cv2(ret, down)
        if up != 1:
            ret = upsample_cv2(ret, up)
        return ret
    
    def convolve_cv2(src, kernel, Ox=0, Oy=0, Width=None, Height=None):
        type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
        if Ox != 0 or Oy != 0 or Height != len(src) or Width != len(src[0]):
            src = src[Oy:Oy+Width, Ox:Ox+Height]
        ret = cv2.filter2D(src, -1, kernel)
        return np.array(ret, dtype=src.dtype)
    
    def blur_cv2(src, ksize, Ox=0, Oy=0, Width=None, Height=None):
        return convolve_cv2(src, get_blur_kernel(ksize), Ox, Oy, Width, Height)
    
    def locmax_cv2(src, ksize, Ox=0, Oy=0, Width=None, Height=None):
        return convolve_cv2(src, get_locmax_kernel(ksize), Ox, Oy, Width, Height)
    
except ImportError:
    _HAS_CV2 = False


try:
    import PIL.Image
    
    _HAS_PIL = True
    
    def _scale_pil(src, factor, Ox=0, Oy=0, Width=None, Height=None):
        type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
        if Ox != 0 or Oy != 0 or Height != len(src) or Width != len(src[0]):
            src = src[Oy:Oy+Width, Ox:Ox+Height]
        shape = (int(Height*factor), int(Width*factor))
        dst = np.array(PIL.Image.fromarray(src).resize(shape, resample=PIL.Image.BILINEAR))
        return dst
    
    def upsample_pil(src, factor, Ox=0, Oy=0, Width=None, Height=None):
        return _scale_pil(src, factor, Ox, Oy, Width, Height)
    
    def downsample_pil(src, factor, Ox=0, Oy=0, Width=None, Height=None):
        return _scale_pil(src, 1.0 / factor, Ox, Oy, Width, Height)
    
    def resample_pil(src, down=4, up=None):
        if up is None:
            up = down
        ret = src
        if down != 1:
            ret = downsample_pil(ret, down)
        if up != 1:
            ret = upsample_pil(ret, up)
        return ret
    
except ImportError:
    _HAS_PIL = False


def upsample_acv(src, factor, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
    stride_src = len(src[0]) * (bpp // 8)
    stride_dst = Width * factor * (bpp // 8)
    dst = np.zeros((Height*factor, Width*factor), dtype=type)
    params = [dst.data, stride_dst, src.data, stride_src, Ox, Oy, Width, Height, bpp, factor]
    w, h = acv.upsample(*params)
    return dst


def downsample_acv(src, factor, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
    stride_src = len(src[0]) * (bpp // 8)
    stride_dst = (Width // factor) * (bpp // 8)
    dst = np.zeros((Height//factor, Width//factor), dtype=type)
    params = [dst.data, stride_dst, src.data, stride_src, Ox, Oy, Width, Height, bpp, factor]
    w, h = acv.downsample(*params)
    return dst


def convolve_acv(src, kernel, Ox=0, Oy=0, Width=None, Height=None):
    type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
    stride_src = len(src[0]) * (bpp // 8)
    stride_dst = Width * (bpp // 8)
    dst = np.zeros((Height, Width), dtype=type)
    if kernel.dtype != np.float64:
        kernel = np.array(kernel, dtype=np.float64)
    params = [dst.data, stride_dst, src.data, stride_src, Ox, Oy, Width, Height, bpp, 
              kernel.data, len(kernel[0]), len(kernel)]
    w, h = acv.convolve(*params)
    return dst

## @brief - Функция увеличения изображения
# @param src - исходное изображение
# @param factor - кратность увеличения (целое число)
# @return - Вернет изображение
def upsample(src, factor, Ox=0, Oy=0, Width=None, Height=None):
    if PREFER_CV2_UPSAMPLE and _HAS_CV2:
        return upsample_cv2(src, factor, Ox, Oy, Width, Height)
    
    type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
    ret = src
    while factor % 4 == 0:
        ret = upsample_acv(ret, 4, Ox, Oy, Width, Height)
        Ox, Oy, Width, Height = 0, 0, None, None
        factor = factor // 4
    while factor % 2 == 0:
        ret = upsample_acv(ret, 2, Ox, Oy, Width, Height)
        Ox, Oy, Width, Height = 0, 0, None, None
        factor = factor // 2
    if factor != 1:
        ret = upsample_acv(ret, factor, Ox, Oy, Width, Height)
    return ret


## @brief - Функция уменьшения изображения
# @param src - исходное изображение
# @param factor - кратность уменьшения (целое число)
# @return - Вернет изображение
def downsample(src, factor, Ox=0, Oy=0, Width=None, Height=None):
    if PREFER_CV2_DOWNSAMPLE and _HAS_CV2:
        return downsample_cv2(src, factor, Ox, Oy, Width, Height)
    
    type, bpp, Ox, Oy, Width, Height = __checkParams(src, Ox, Oy, Width, Height)
    ret = src
    while factor % 4 == 0:
        ret = downsample_acv(ret, 4, Ox, Oy, Width, Height)
        Ox, Oy, Width, Height = 0, 0, None, None
        factor = factor // 4
    while factor % 2 == 0:
        ret = downsample_acv(ret, 2, Ox, Oy, Width, Height)
        Ox, Oy, Width, Height = 0, 0, None, None
        factor = factor // 2
    if factor != 1:
        ret = downsample_acv(ret, factor, Ox, Oy, Width, Height)
    return ret

## @brief - Функция масштабирования изображения (оптимизированная)
# @param src - исходное изображение
# @param down - кратность уменьшения (целое число)
# @param up - кратность увеличения (целое число)
# @return - Вернет изображение
def resample(src, down=4, up=None):
    if up is None:
        up = down
    ret = src
    if down != 1:
        ret = downsample(ret, down)
    if up != 1:
        ret = upsample(ret, up)
    return ret


def convolve(src, kernel, Ox=0, Oy=0, Width=None, Height=None):
    if PREFER_CV2_CONVOLVE and _HAS_CV2:
        return convolve_cv2(src, kernel, Ox, Oy, Width, Height)
    return convolve_acv(src, kernel, Ox, Oy, Width, Height)


def blur(src, ksize, Ox=0, Oy=0, Width=None, Height=None):
    return convolve(src, get_blur_kernel(ksize), Ox, Oy, Width, Height)


def locmax(src, ksize, Ox=0, Oy=0, Width=None, Height=None):
    return convolve(src, get_locmax_kernel(ksize), Ox, Oy, Width, Height)



## @brief - Функция вычисления шумовой маски изображения
# @param src - серия исходных изображений (list [])
# @param Ox - отступ по X
# @param Oy - отступ по Y
# @param Width - широна изображения
# @param Height - высота изображения
# @return - маску изображения
def getNoiseMask(src, Ox=0, Oy=0, Width=None, Height=None, thresh=3.):
    #type, bpp, Ox, Oy, Width, Height = __checkParams(src[0], Ox, Oy, Width, Height)
    sumImg = np.zeros(np.shape(src[0]), dtype=np.float64)
    for i in src:
        sumImg += i
    avgImg = sumImg / len(src)

    tmp = np.zeros(np.shape(src[0]))
    for i in src:
        tmp += (i - avgImg) ** 2
    sigma = np.sqrt(tmp / len(src))

    avg = np.average(sigma)
    maxLevel = avg * thresh

    mask = np.ones(np.shape(src[0]), dtype=np.uint8)
    for y in range(0, len(avgImg)):
        for x in range(0, len(avgImg[0])):
            mask[y][x] = 0 if sigma[y][x] > maxLevel else 1

    return mask


## @brief - Функция вычисления градиентной маски изображения
# @param src - серия исходных изображений (list [])
# @param Ox - отступ по X
# @param Oy - отступ по Y
# @param Width - широна изображения
# @param Height - высота изображения
# @return - маску изображения
def getGradientMask(src, Ox=0, Oy=0, Width=None, Height=None, thresh=3.):
    #type, bpp, Ox, Oy, Width, Height = __checkParams(src[0], Ox, Oy, Width, Height)
    sumImg = np.zeros(np.shape(src[0]), dtype=np.float64)
    for i in src:
        sumImg += i
    avgImg = sumImg / len(src)

    grad = np.ones(np.shape(src[0]))
    for y in range(1, len(avgImg)):
        for x in range(1, len(avgImg[0])):
            ns = avgImg[y-1][x-1] + avgImg[y-1][x] - avgImg[y][x-1] - avgImg[y][x]
            ew = avgImg[y-1][x-1] + avgImg[y][x-1] - avgImg[y-1][x] - avgImg[y][x]
            grad[y][x] = abs(ns) + abs(ew)

    grad[0] = grad[1]
    for i in range(1, len(grad)):
        grad[i][0] = grad[i][1]

    avg = np.average(grad)
    maxLevel = avg * thresh

    mask = np.ones(np.shape(src[0]), dtype=np.uint8)
    for y in range(0, len(avgImg)):
        for x in range(0, len(avgImg[0])):
            mask[y][x] = 0 if grad[y][x] > maxLevel else 1

    return mask

## @brief - Функция объединения масок изображения
# @param mask1 - первая маска
# @param mask2 - вторая маска
# @return - маску изображения
def getUnionMask(mask1, mask2):
    return np.minimum(mask1, mask2)

if __name__ == '__main__':
    import cv2
    import numpy as np
    from astrocv.ganerate import *
    from astrocv.search import *

    Noise = 30
    dNoise = 59
    imgList = []
    for i in range(10):
            img = np.zeros((600, 600), dtype=np.uint8)
            addNoiseNorm(img, Noise, dNoise)
            drawObject(img, 400, 400, 10)
            drawObject(img, 450, 450, 15)
            imgList.append(img)
    grad = getGradientMask(imgList, thresh=1.5)
    nois = getNoiseMask(imgList)
    res = np.minimum(grad, nois)
    while 1:
        cv2.imshow("1", grad/grad.max())
        cv2.imshow("2", nois/nois.max())
        cv2.imshow("3", res / res.max())
        val = cv2.waitKey(20)
        if val & 0xff == ord('q'):
            break
        if val & 0xff == ord('w'):
            exit(0)

    resol = [(400, 400), (480, 640), (600, 800), (768, 1024), (1024, 1280), (2048, 2048)]
    type = [np.uint8, np.uint16, np.int32]
    TESTS = [
        {"src": np.zeros(resol[0], dtype=type[0]),
         "width": resol[0][0],
         "height": resol[0][1],
         "bpp": 8,
         "method": METHOD_BEST_INTEGRAL,
         "minSize": 2,
         "maxSize": 20,
         "MinCertainty": 0.05,
         "NMaxObjects": 1},

        {"src": np.zeros(resol[1], dtype=type[1]),
         "width": resol[1][0],
         "height": resol[1][1],
         "bpp": 16,
         "method": METHOD_BEST_INTEGRAL,
         "minSize": 2,
         "maxSize": 20,
         "MinCertainty": 0.05,
         "NMaxObjects": 5},

        {"src": np.zeros(resol[2], dtype=type[2]),
         "width": resol[2][0],
         "height": resol[2][1],
         "bpp": 32,
         "method": METHOD_BEST_INTEGRAL,
         "minSize": 2,
         "maxSize": 20,
         "MinCertainty": 0.05,
         "NMaxObjects": 10},

        {"src": np.zeros(resol[3], dtype=type[0]),
         "width": resol[3][0],
         "height": resol[3][1],
         "bpp": 8,
         "method": METHOD_BEST_CONTRAST,
         "minSize": 2,
         "maxSize": 20,
         "MinCertainty": 1.1,
         "NMaxObjects": 1},

        {"src": np.zeros(resol[4], dtype=type[1]),
         "width": resol[4][0],
         "height": resol[4][1],
         "bpp": 16,
         "method": METHOD_BEST_CONTRAST,
         "minSize": 2,
         "maxSize": 20,
         "MinCertainty": 1.1,
         "NMaxObjects": 5},

        {"src": np.zeros(resol[5], dtype=type[2]),
         "width": resol[5][0],
         "height": resol[5][1],
         "bpp": 32,
         "method": METHOD_BEST_CONTRAST,
         "minSize": 2,
         "maxSize": 20,
         "MinCertainty": 5.1,
         "NMaxObjects": 10},
        ]

    bpp = (8, 16, 32)
    method = (METHOD_BEST_INTEGRAL, METHOD_BEST_CONTRAST)
    Noise = 30
    dNoise = 7
    minSize = 4
    maxSize = 18
    MinCertainty = 1
    NMaxObjects = 10
    import time

    imgList = []
    for j in range(0, len(bpp)):
        for i in resol:
            img = np.zeros(i, dtype=type[j])
            addNoiseNorm(img, Noise, dNoise)
            drawObject(img, 100, 100, 4)
            drawObject(img, 200, 200, 20)
            drawObject(img, 300, 300, 15)
            imgList.append(img)

    for m in method:
        for j in range(0, len(bpp)):
            for i in range(0, len(resol)):
                r = [{}]
                count = 0
                t0 = time.time()
                while time.time() < t0 + 1:
                    r = searchObjects(imgList[i + len(resol) * j], imgList[i + len(resol) * j], m, minSize, maxSize, MinCertainty, NMaxObjects)
                    count += 1
                print("bpp: ", bpp[j], " resol:", (len(imgList[i + len(resol) * j][0]), len(imgList[i + len(resol) * j])), "method", m, "count", count)
                #for k in range(0, len(r)):
                #    selectObject(imgList[i + len(resol) * j], r[k]["X"], r[k]["Y"], r[k]["Left"], r[k]["Top"], r[k]["Right"], r[k]["Bottom"], bpp[j])
                print("len(result)", len(r))
                #for i in r:
                #    print(i)
                #print()
                cv2.imshow("result", imgList[i+len(resol) * j])
                cv2.waitKey(1)



    maxSignals = ((1 << 8)-1, (1 << 16)-1, (1 << 16)-1)
    for j in range(0, len(bpp)):
        for i in range(0, len(resol)):
            img1 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            img2 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            img1 = np.reshape(np.frombuffer(img1, dtype=type[j]),
                              (resol[i][0], resol[i][1]))
            img2 = np.reshape(np.frombuffer(img2, dtype=type[j]),
                              (resol[i][0], resol[i][1]))
            res = add(img1, img2)
            res = np.reshape(np.frombuffer(res, dtype=type[j]), (resol[i][0], resol[i][1]))
            cv2.imshow("result", res)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    for j in range(0, len(bpp)):
        for i in range(0, len(resol)):
            img1 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            img1 = np.reshape(np.frombuffer(img1, dtype=type[j]),
                            (resol[i][0], resol[i][1]))
            res = contrast(img1, 2, 5)
            res = np.reshape(np.frombuffer(res, dtype=type[j]), (resol[i][0], resol[i][1]))
            cv2.imshow("result", res)
            cv2.waitKey(1)

    for j in range(0, len(bpp)):
        for i in range(0, len(resol)):
            img1 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            img2 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            img1 = np.reshape(np.frombuffer(img1, dtype=type[j]),
                            (resol[i][0], resol[i][1]))
            img2 = np.reshape(np.frombuffer(img2, dtype=type[j]),
                            (resol[i][0], resol[i][1]))
            res = difference(img1, img2)
            res = np.reshape(np.frombuffer(res, dtype=type[j]), (resol[i][0], resol[i][1]))
            cv2.imshow("result", res)
            cv2.waitKey(1)

    for j in range(0, len(bpp)):
        for i in range(0, len(resol)):
            img1 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            img1 = np.reshape(np.frombuffer(img1, dtype=type[j]),
                            (resol[i][0], resol[i][1]))
            res = smooth(img1, 3)
            res = np.reshape(np.frombuffer(res, dtype=type[j]), (resol[i][0], resol[i][1]))
            cv2.imshow("result", res)
            cv2.waitKey(1)


    imgList = []
    for j in range(0, len(bpp)):
        for i in range(0, len(resol)):
            imgList.append(np.reshape(np.frombuffer(np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j]), dtype=type[j]),
                            (resol[i][0], resol[i][1])))

    for j in range(0, len(bpp)):
        for i in range(0, len(resol)):
            #img1 = np.random.randint(0, maxSignals[j], resol[i][0] * resol[i][1], dtype=type[j])
            #img1 = np.reshape(np.frombuffer(img1, dtype=type[j]),
            #                (resol[i][1], resol[i][0]))
            img1 = imgList[i + j * len(resol)]
            img_c = calibrationFrom(img1, 20)
            img_c = np.reshape(np.frombuffer(img_c, dtype=type[j]),
                            (resol[i][0], resol[i][1]))
            res = applyCalibration(img1, img_c)
            res = np.reshape(np.frombuffer(res, dtype=type[j]), (resol[i][0], resol[i][1]))
            cv2.imshow("result", res)
            cv2.waitKey(1)

