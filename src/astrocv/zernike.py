#!/usr/bin/python
# -*- coding: utf-8 -*-
## @file zernike.py
# @brief Class Zernike polynomials

import numpy as np
import cv2

polinomials = dict()

## @brief - generate polinomials Zernike
# @param size - polinomials size
def generatePolinomialsZernike(size=200):

    zernikeBasis = {}
    ZERNIKE = [
        (0, 0, "Piston", lambda rho, theta: 1.0),
        (1, -1, "Tilt Y", lambda rho, theta: 2.0 * rho * np.sin(theta)),
        (1, 1, "Tilt X", lambda rho, theta: 2.0 * rho * np.cos(theta)),
        (2, -2, "Astigmatism 45", lambda rho, theta: np.sqrt(6.0) * rho ** 2 * np.sin(2.0 * theta)),
        (2, 0, "Defocus", lambda rho, theta: np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)),
        (2, 2, "Astigmatism 90", lambda rho, theta: np.sqrt(6.0) * rho ** 2 * np.cos(2.0 * theta)),
        (3, -3, "", lambda rho, theta: np.sqrt(8.0) * rho ** 3 * np.sin(3.0 * theta)),
        (3, -1, "Coma Y", lambda rho, theta: np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.sin(theta)),
        (3, 1, "Coma X", lambda rho, theta: np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.cos(theta)),
        (3, 3, "", lambda rho, theta: np.sqrt(8.0) * rho ** 3 * np.cos(3.0 * theta)),
        (4, -4, "", lambda rho, theta: np.sqrt(10.0) * rho ** 4 * np.sin(4.0 * theta)),
        (4, -2, "Secondary astigmatism m", lambda rho, theta: np.sqrt(10.0) *
                                                              (4 * rho ** 3 - 3 * rho ** 2) * np.sin(2.0 * theta)),
        (4, 0, "Spherical aberration, defocus", lambda rho, theta: np.sqrt(5.0) *
                                                                   (6 * rho ** 4 - 6 * rho ** 2 + 1)),
        (4, 2, "Secondary astigmatism m", lambda rho, theta: np.sqrt(10.0) *
                                                             (4 * rho ** 4 - 3 * rho ** 2) * np.cos(2.0 * theta)),
        (4, 4, "", lambda rho, theta: np.sqrt(10.0) * rho ** 4 * np.cos(4.0 * theta))
    ]

    for i in ZERNIKE:
        zernikeBasis[(size, i[0], i[1])] = np.zeros((size, size))

    x0 = y0 = size / 2
    radius = size / 2
    for row in range(size):
        y = float(row - y0) / float(radius)
        for col in range(size):
            x = float(col - x0) / float(radius)
            rho = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            for i in range(0, len(ZERNIKE)):
                zernikeBasis[(size,
                            ZERNIKE[i][0],
                            ZERNIKE[i][1])][row, col] = ZERNIKE[i][3](rho, theta)

    polinomials[size] = zernikeBasis


## @brief - generate near zone
# @param size - polinomial size
# @param coefs - list polinomial coefficients
# @param powerIn - power at the input
# @param r0 - radius blind zone (0.0 .. 1.0)
# @return numpy.array: (amplitude, phase)
def makeNearZone(size, coefs, powerIn, r0=0.0):

    amplitude = np.zeros((size, size))
    if r0 > 1.0:
        r0 = 1.0

    if size not in polinomials:
        generatePolinomialsZernike(size)
    zernikeBasis = polinomials[size]

    x0 = y0 = size / 2
    radius = size / 2
    for row in range(size):
        y = float(row - y0) / float(radius)
        for col in range(size):
            x = float(col - x0) / float(radius)
            rho = np.sqrt(x ** 2 + y ** 2)
            if rho < 1.0 and rho > r0:
                amplitude[row, col] = np.sqrt(powerIn)

    if len(coefs) < len(zernikeBasis):
        for i in range(0, len(zernikeBasis) - len(coefs)):
            coefs.append(0.0)
        #raise Exception("len(coefs) < len(self.zernikeBasis): {}".format(len(coefs)))

    sum = np.zeros((size, size))
    index = 0
    for i in zernikeBasis:
        sum += coefs[index] * zernikeBasis[i]
        index += 1
    return (amplitude, sum)


## @brief - generate far zone
# @param nearAmpl - numpy.array - amplitude in near zone
# @param nearPhase - numpy.array - phase in near zone
# @param diameter - diameter aperture
# @param wavelength - wave length
# @param pixSize - size one pixel
# @param focus - focus length
# @param exposition - exposition
# @param electronWell - potencial electron well
# @param quantumEffect - quantum effectiveness
# @return - numpy.array - image in far zone
def makeFarZone(nearAmpl, nearPhase, diameter=0.2, wavelength=0.5e-6,
                pixSize=5e-6, focus = 2.0, exposition=2.0, electronWell=8000, quantumEffect=1.0):
    h = 6.626176e-34
    c = 3.0e11

    #energy = nearAmpl*np.pi*((diameter**2)/4)*exposition
    #quants = energy / (h*c/wavelength)
    #print(quants.max())

    size = len(nearAmpl)
    near = nearAmpl * ((diameter/size)**2) * np.exp(-1j * 2.0 * np.pi * nearPhase)

    far = np.fft.fft2(near)
    far = np.fft.fftshift(far)
    far = abs(far ** 2)

    pixFarZone = (focus * wavelength) / diameter

    attitude = pixSize / pixFarZone
    #print("att:", attitude)
    #far = cv2.resize(far, (int(size // attitude), int(size // attitude)))
    #countPhotons = ((far * (pixFarZone **2) * exposition) / (h * c / wavelength)) * quantumEffect * (attitude **2)
    quantEnergy = (h * c / wavelength)
    pixelScale = (pixSize / focus) / (wavelength / diameter)
    countPhotons = ((far * exposition) / quantEnergy) * quantumEffect * (pixelScale**2)

    fullness = countPhotons / electronWell

    #far = far / far.max()
    one = np.ones((len(far), len(far)))
    fullness = np.minimum(fullness, one)
    far = fullness*255
    far = np.array(far, dtype=np.uint8)

    rIn = wavelength / diameter
    rOut = pixSize / focus
    newSize = rIn/rOut * size
    far = cv2.resize(far, (int(newSize + 0.5), int(newSize + 0.5)), interpolation=cv2.INTER_LINEAR)
    return far


## @brief - generate far zone after gartman detector
# @param nearAmpl - numpy.array - amplitude in near zone
# @param nearPhase - numpy.array - phase in near zone
# @param partsX - count lens in column
# @param partsY - count lens in rows
# @return - numpy.array - image in far zone
def makeGartman(nearAmpl, nearPhase, partsX, partsY):
    near = nearAmpl * np.exp(-1j * 2.0 * np.pi * nearPhase)
    size = len(nearAmpl)
    gart = np.zeros((size, size))

    dx = int(size // partsX)
    dy = int(size // partsY)

    for j in range(0, partsY):
        for i in range(0, partsX):
            tmp = np.zeros((dy, dx))
            for k in range(0, dy):
                tmp[k] = near[j * dy + k][i * dx: i * dx + dx]
            g = np.fft.fft2(tmp)
            g = np.fft.fftshift(g)
            #g[int(dx / 2)][int(dy / 2)] = g.max()
            for k in range(0, dy):
                gart[j * dy + k][i * dx: i * dx + dx] = g[k]

    gart = abs(gart ** 2)
    gart = gart / gart.max()
    return gart


if __name__ == '__main__':
    import cv2
    import numpy as np

    size = 128
    #K = [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    K = [1.0, 0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    ampl, poli = makeNearZone(size, K, 1e-8, r0=0.3)
    far = makeFarZone(ampl, poli)
    #gar = makeGartman(ampl, poli, 8, 8)

    #ampl, poli = zer.makeNearZone(K)
    #far = zer.makeFarZone(ampl, poli)
    print(far.max())
    cv2.imshow('far', far)
    cv2.imshow('ampl', ampl/ampl.max())
    cv2.imshow('poli', poli/poli.max())
    #cv2.imshow('gar', gar / gar.max())
    while True:
        val = cv2.waitKey(1)
        if val & 0xff == ord('q'):
            exit(0)
