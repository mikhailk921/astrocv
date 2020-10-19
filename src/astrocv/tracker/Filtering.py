# -*- coding: utf-8 -*-
## @file KalmanFilter.py
# @brief Файл содержит класс с функциями фильтра Калмана для корректировки и предсказания
import cv2
import numpy as np
import copy
class TrackerObject:
    def __init__(self, x=0.0, y=0.0, intens=0.0, area=0.0, vx=0.0, vy=0.0, timestamp=None, ax=0.0, ay=0.0, keypoint=None):
        if timestamp is None:
            timestamp = time.time()
        self.point = [x, y]
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.timeDetect = timestamp
        self.intens = intens
        self.area = area
        self.source = keypoint



## @brief Класс реализующий фильтр Калмана
class KalmanFilter6x6:
    ## @brief - Конструктор класса
    # @param pt - класс точки (TrackerObject)
    # @param __deltaTime - дельта времени сек
    # @param __accelNoiseMag - шум
    # @param koefMeasurNoiseCov - точность измерений
    def __init__(self, pt, deltaTime=0.2, accelNoiseMag=0.5, koefMeasurNoiseCov=0.1):
        self.initialPoints = []
        self.MIN_INIT_VALS = 4
        self.lastPointResult = pt
        self.initialized = False
        self.__deltaTime = deltaTime
        self.__accelNoiseMag = accelNoiseMag
        self.__koefMeasurNoiseCov = koefMeasurNoiseCov
        self.initialPoints.append(pt)
        self._DP = 6
        self._MP = 4
        self.t0 = 0

        self._statePre = np.zeros(self._DP, dtype='float64')
        self._statePost = np.zeros(self._DP, dtype='float64')
        self._transitionMatrix = np.zeros((self._DP, self._DP), dtype='float64')
        self._processNoiseCov = np.zeros((self._DP, self._DP), dtype='float64')
        self._measurementMatrix = np.zeros((self._MP, self._DP), dtype='float64')
        self._measurementNoiseCov = np.zeros((self._MP, self._MP), dtype='float64')
        self._errorCovPost = np.zeros((self._DP, self._DP), dtype='float64')
        self._errorCovPre = np.zeros((self._DP, self._DP), dtype='float64')

        self._temp1 = np.zeros((self._DP, self._DP), dtype='float64')
        self._temp2 = np.zeros((self._MP, self._DP), dtype='float64')
        self._temp3 = np.zeros((self._MP, self._MP), dtype='float64')
        self._temp4 = np.zeros((self._MP, self._DP), dtype='float64')
        self._gain = np.zeros((self._DP, self._MP), dtype='float64')
        self._temp5 = np.zeros(self._MP, dtype='float64')

    def _get_lin_regress(self, in_data, start_pos, in_data_size, xy):
        m1 = m2 = m3_x = m4_x = m3_y = m4_y = 0.0
        m3_i = m4_i = m3_s = m4_s = 0.0
        el_count = in_data_size - start_pos
        for i in range(start_pos, in_data_size):
            m1 += i
            m2 += i * i

            m3_x += in_data[i].point[0]
            m4_x += i * in_data[i].point[0]

            m3_y += in_data[i].point[1]
            m4_y += i * in_data[i].point[1]

            m3_i += in_data[i].intens
            m4_i += i * in_data[i].intens

            m3_s += in_data[i].area
            m4_s += i * in_data[i].area
        det_1 = 1 / (el_count * m2 - m1 * m1)
        m1 *= -1
        xy["kx"] = det_1 * (m1 * m3_x + el_count * m4_x)
        xy["bx"] = det_1 * (m2 * m3_x + m1 * m4_x)

        xy["ky"] = det_1 * (m1 * m3_y + el_count * m4_y)
        xy["by"] = det_1 * (m2 * m3_y + m1 * m4_y)

        xy["ki"] = det_1 * (m1 * m3_i + el_count * m4_i)
        xy["bi"] = det_1 * (m2 * m3_i + m1 * m4_i)

        xy["ks"] = det_1 * (m1 * m3_s + el_count * m4_s)
        xy["bs"] = det_1 * (m2 * m3_s + m1 * m4_s)

    ## @brief - функция инициализации фильта Калмана
    # @param startPoint - стартовая точка
    def _CreateLinear(self, startPoint):
        self._transitionMatrix = np.array(([1, 0, 0, 0, self.__deltaTime, 0],
                                           [0, 1, 0, 0, 0, self.__deltaTime],
                                           [0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]), dtype='float64')

        # init...
        self.lastPointResult = startPoint

        self._statePre = np.array((startPoint.point[0], startPoint.point[1],
                                   startPoint.intens, startPoint.area,
                                   startPoint.vx, startPoint.vy), dtype='float64')

        self._statePost = np.array((startPoint.point[0], startPoint.point[1],
                                    startPoint.intens, startPoint.area,
                                    0, 0), dtype='float64')

        n1 = (self.__deltaTime ** 4.) / 4.
        n2 = (self.__deltaTime ** 3.) / 2.
        n3 = self.__deltaTime ** 2.

        self._processNoiseCov = np.array(([n1, 0, 0, 0, n2, 0],
                                          [0, n1, 0, 0, 0, n2],
                                          [0, 0, n1, 0, 0, 0],
                                          [0, 0, 0, n1, 0, 0],
                                          [n2, 0, 0, 0, n3, 0],
                                          [0, n2, 0, 0, 0, n3]),
                                         dtype='float64')

        self._processNoiseCov *= self.__accelNoiseMag

        self._measurementMatrix = np.array(([1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0]), dtype='float64')

        k = self.__koefMeasurNoiseCov
        self._measurementNoiseCov = np.array(([k, 0, 0, 0],
                                              [0, k, 0, 0],
                                              [0, 0, k, 0],
                                              [0, 0, 0, k]), dtype='float64')

        self._errorCovPost = np.array(([0.1, 0, 0, 0, 0, 0],
                                       [0, 0.1, 0, 0, 0, 0],
                                       [0, 0, 0.1, 0, 0, 0],
                                       [0, 0, 0, 0.1, 0, 0],
                                       [0, 0, 0, 0, 0.1, 0],
                                       [0, 0, 0, 0, 0, 0.1]), dtype='float64')

        self.initialized = True


    ## @brief - функция изменяет текущее значение deltaTime
    # @param dt - значение deltaTime
    # @return - Вернет новое значение deltaTime
    def resetDeltaTime(self, dt):
        self.__deltaTime = dt
        self._transitionMatrix = np.array(([1, 0, 0, 0, self.__deltaTime, 0],
                                           [0, 1, 0, 0, 0, self.__deltaTime],
                                           [0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]), dtype='float64')
        n1 = (self.__deltaTime ** 4.) / 4.
        n2 = (self.__deltaTime ** 3.) / 2.
        n3 = self.__deltaTime ** 2.
        self._processNoiseCov = np.array(([n1, 0, 0, 0, n2, 0],
                                          [0, n1, 0, 0, 0, n2],
                                          [0, 0, n1, 0, 0, 0],
                                          [0, 0, 0, n1, 0, 0],
                                          [n2, 0, 0, 0, n3, 0],
                                          [0, n2, 0, 0, 0, n3]),
                                         dtype='float64')
        return self.__deltaTime


    ## @brief - функция предсказания
    # @param dt - значение deltaTime
    # @return - Вернет TrackerObject
    def GetPointPrediction(self, dt):
        try:
            self.resetDeltaTime(dt)
            if self.initialized:
                prediction = self._predict()
                prediction = TrackerObject(prediction[0], prediction[1],#x, y
                                                                  prediction[2], prediction[3],#intens, area
                                                                  prediction[4], prediction[5],#vx, vy
                                                                  self.t0 + dt)                #timeDetect
            else:
                prediction = self.lastPointResult
            return prediction

        except Exception as e:
            print("Kalman:GetPointPredict", e)

    ## @brief - функция обновления фильта Калмана
    # @param pt - координаты точки
    # @return - Вернет TrackerObject
    def Update(self, pt):
        try:
            self.t0 = pt.timeDetect
            if not self.initialized:
                if len(self.initialPoints) < self.MIN_INIT_VALS:
                    self.initialPoints.append(copy.deepcopy(pt))
                if len(self.initialPoints) == self.MIN_INIT_VALS:
                    xy = {"kx": 0, "bx": 0, "ky": 0, "by": 0, "intens": 0, "area": 0}
                    self._get_lin_regress(self.initialPoints, 0, self.MIN_INIT_VALS, xy)

                    self._CreateLinear(TrackerObject(xy["kx"] * (self.MIN_INIT_VALS - 1) + xy["bx"],
                                        xy["ky"] * (self.MIN_INIT_VALS - 1) + xy["by"],
                                        xy["ki"] * (self.MIN_INIT_VALS - 1) + xy["bi"],
                                        xy["ks"] * (self.MIN_INIT_VALS - 1) + xy["bs"],
                                        xy["kx"], xy["ky"], pt.timeDetect))

            if self.initialized:

                measurement = np.array((pt.point[0], pt.point[1],
                                        pt.intens, pt.area), dtype='float64')

                estimated = self._correct(measurement)

                self.lastPointResult.point[0] = estimated[0]
                self.lastPointResult.point[1] = estimated[1]
                self.lastPointResult.intens = estimated[2]
                self.lastPointResult.area = estimated[3]
                self.lastPointResult.timeDetect = pt.timeDetect
            else:
                self.lastPointResult = copy.deepcopy(pt)
            return self.lastPointResult
        except Exception as e:
            print("Kalman:Update", e)

    ## @brief - функция реализующая предсказание
    def _predict(self):
        try:
            # обновляем состояние: x '(k) = A * x (k)
            self._statePre = np.dot(self._transitionMatrix, self._statePost)

            # transitionMatrix[6][6] * errorCovPost[6][6]
            self._temp1 = np.dot(self._transitionMatrix, self._errorCovPost)

            # P '(k) = temp1_T * transitionMatrix (A) + processNoiseCov_T (Q)
            # temp1 to temp1_T
            temp1_T = self._temp1.transpose()

            # temp1_T * A
            temp12 = np.dot(temp1_T, self._transitionMatrix)

            # processNoiseCov to processNoiseCov_T
            processNoiseCov_T = self._processNoiseCov.transpose()

            # temp12 + Q_T
            self._errorCovPre = temp12 + processNoiseCov_T

            self._statePost = self._statePre
            self._errorCovPost = self._errorCovPre

            return self._statePre
        except Exception as e:
            print("Kalman:predict", e)
            return self._statePre

    ## @brief - функция реализующая коррекцию
    # @param xy - координаты точки
    def _correct(self, xy):
        try:
            z = np.array((xy[0], xy[1], xy[2], xy[3]), dtype='float64')

            # temp2 =  measurementMatrix[4][6] (H) * errorCovPre[6][6] (P'(k))
            self._temp2 = np.dot(self._measurementMatrix, self._errorCovPre)

            # temp3 = temp2 * measurementMatrix_T (Ht) + measurementNoiseCov (R)
            # measurementMatrix to measurementMatrix_T
            measurementMatrix_T = self._measurementMatrix.transpose()

            # temp2 * H_T
            temp21 = np.dot(self._temp2, measurementMatrix_T)

            # temp21_T + measurementNoiseCov (R)

            self._temp3 = temp21 + self._measurementNoiseCov

            # temp4 = inv (temp3) * temp2 = Kt (k)
            # inv(temp3)
            temp3_inv = np.linalg.inv(self._temp3)

            # temp4 = inv (temp3) * temp2
            self._temp4 = np.dot(temp3_inv, self._temp2)

            # gain = temp4_T
            self._gain = self._temp4.transpose()

            # temp5 =  z(k) - measurementMatrix[4][6] (H) * statePre[6][1] (x'(k))
            # measurementMatrix[4][6] (H) * statePre[6] (x'(k))
            temp52 = np.dot(self._measurementMatrix, self._statePre)

            # temp5 =  z(k) + temp52 * -1
            self._temp5 = z - temp52

            # statePost (x(k)) = statePre (x'(k)) +  gain[4][6](K (k)) * temp5[4]
            # gain[4][6](K (k)) * temp5[4]
            temp53 = np.dot(self._gain, self._temp5)

            # statePre[4] (x'(k)) +  temp53[2]
            self._statePost = self._statePre + temp53

            # errorCovPost P(k) = errorCovPre P'(k) - gain K(k) * temp2
            # gain K(k) * temp2
            temp22 = np.dot(self._gain, self._temp2)

            # errorCovPre P'(k) + temp22 * -1
            self._errorCovPost = self._errorCovPre - temp22

            return self._statePost
        except Exception as e:
            print("Kalman:correct", e)
            return self._statePost


## @brief Класс фильтрации методом наименьших квадратов
class MNK:
    ## @brief - Конструктор класса
    # @param trace - ссылка на трек
    # @param point - начальная точка (TrackerObject)
    def __init__(self, trace, point):
        self.MIN_INIT_VALS = 4
        self.__degree = 2
        self.polyX = None
        self.polyY = None
        self.t0 = 0
        self.trace = trace
        self.lastPointResult = point

    ## @brief - функция расчёта моментов времени трека, последний будет равен 0
    # @param point - последняя точка
    # @return - Вернет список
    def calcTime(self, point):
        self.t0 = point.timeDetect
        dtime = []
        for i in self.trace:
            dtime.append(i.timeDetect - self.t0)
        dtime.append(point.timeDetect - self.t0)
        return dtime

    ## @brief - функция расчёта положений объекта
    # @param point - последняя точка
    # @return - Вернет списоки x, y
    def getPoints(self, point):
        x, y = [], []
        for i in self.trace:
            x.append(i.point[0])
            y.append(i.point[1])
        x.append(point.point[0])
        y.append(point.point[1])
        return x, y


    ## @brief - функция обновления
    # @param point - координаты новой точки
    # @return - Вернет TrackerObject
    def Update(self, point):
        try:
            if len(self.trace) < 4:
                return point
            dtime = self.calcTime(point)
            rangeX, rangeY = self.getPoints(point)
            self.polyX = np.polyfit(dtime, rangeX, self.__degree)
            self.polyY = np.polyfit(dtime, rangeY, self.__degree)

            x = np.polyval(self.polyX, 0)
            y = np.polyval(self.polyY, 0)

            return TrackerObject(x, y, point.intens, point.area,
                                 self.polyX[1], self.polyY[1], point.timeDetect,
                                 ax=self.polyX[0], ay=self.polyY[0])

        except Exception as e:
            print("MNK: Update", e)

    ## @brief - функция предсказания
    # @param dt - delta time
    # @return - Вернет TrackerObject
    def GetPointPrediction(self, dt):
        if self.polyX is None or self.polyY is None:
            return self.lastPointResult

        x = np.polyval(self.polyX, dt)
        y = np.polyval(self.polyY, dt)

        return TrackerObject(x, y, self.lastPointResult.intens, self.lastPointResult.area,
                             self.polyX[1], self.polyY[1], self.t0 + dt,
                             ax=self.polyX[0], ay=self.polyY[0])


    ## @brief - функция предсказания на момент времени
    # @param dt - delta time
    # @return - Вернет TrackerObject
    def GetPointPredictionForTime(self, time):
        return self.GetPointPrediction(time - self.t0)


if __name__ == '__main__':
    import time
    import numpy as np


    poly = [0.0001, 1, 10]
    point = np.polyder(poly, 0)
    vel = np.polyder(poly, 1)
    accum = np.polyder(poly, 2)
    print(point, vel, accum)
    p = np.polyval(poly, 5)
    print(p)
    pass


    class TrackerObject:
        def __init__(self, x=0, y=0, vx=0, vy=0, intens=100, area=10):
            self.point = (x, y)
            self.vx = vx
            self.vy = vy
            self.timeDetect = time.time()
            self.intens = intens
            self.area = area


    try:

        n = np.array((0, 0))
        print(n)

        kalman = KalmanFilter6x6(TrackerObject(1, 1))
        k = 1
        while 1:

            last = kalman.GetPointPrediction()
            # print(last)
            last = kalman.Update({"x": k, "y": k}, True)
            print(k, last)
            k += 1
            if k == 50:
                continue
    except Exception as e:
        print(e)
