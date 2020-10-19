#!/astrocv/test/test_objects_detect
# -*- coding: utf-8 -*-

import astrocv as acv
import cv2
import numpy as np



#maxsize = 640 * 480
width = 800
height = 800
bpp = 16
type = {8: np.uint8, 16: np.uint16, 32: np.int32}[bpp]
maxSignal = 200


Noise = 30
dNoise = 7
method = acv.search.METHOD_BEST_CONTRAST
minSize = 3
MaxSize = 12
MinCertainty = 1.0

scale = {8: 1, 16: 256, 32: 256}[bpp]
maxSignal *= scale
Noise *= scale
dNoise *= scale

recX = 10
recY = 10
rectCounts = recX * recY

startX = []
startY = []
widthROI = []
heightROI = []

for j in range(0, recY):
    for i in range(0, recX):
        startX.append(50 + i * 50)
        startY.append(50 + j * 50)
        widthROI.append(50)
        heightROI.append(50)

roi = np.zeros((rectCounts * 4), dtype=np.int32)

ROI = bytearray()
for i in range(0, rectCounts):
    #ROI += struct.pack("iiii", startX[i], startY[i], widthROI[i], heightROI[i])
    roi[0 + i * 4] = startX[i]
    roi[1 + i * 4] = startY[i]
    roi[2 + i * 4] = widthROI[i]
    roi[3 + i * 4] = heightROI[i]

mem = memoryview(ROI)

while True:
    img1 = np.zeros((height, width), dtype=type)
    acv.addNoiseNorm(img1, Noise, dNoise)

    for i in range(0, rectCounts):
        acv.drawObject(img1, int(roi[0 + i * 4] + roi[2 + i * 4]/2), int(roi[1 + i * 4] + roi[3 + i * 4]/2), 10)

    r = acv.searchObjectsForMultiROI(rectCounts, roi, img1, img1, method, minSize, MaxSize, MinCertainty)
    
    for i in range(0, rectCounts):
        cv2.rectangle(img1, (startX[i], startY[i]), (startX[i] + widthROI[i], startY[i] + heightROI[i]), maxSignal)
        #print(r[i]["X"], r[i]["Y"])
    for i in range(0, len(r)):
        if True: # r[i]["X"] != 0 and r[i]["Y"] != 0:
            acv.addMarker(img1, r[i]["X"] + startX[i], r[i]["Y"] + startY[i], r[i]["Left"] + startX[i], r[i]["Top"]
                          + startY[i], r[i]["Right"] + startX[i], r[i]["Bottom"] + startY[i])


    cv2.imshow("1", img1)

    val = cv2.waitKey(1)
    if val & 0xff == ord('q'):
        exit(0)
