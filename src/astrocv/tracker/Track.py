# -*- coding: utf-8 -*-
# @file Track.py
# @brief Файл содержит класс описывающий трек
import astrocv.tracker.Filtering as filter
import copy
import math
import time


## @brief Класс трека
class Track:
    ## @brief - Конструктор класса
    # @param object - класс точки (TrackerObject)
    # @param trackID - номер трека
    # @param deltaTime - дельта времени
    # @param accelNoiseMag - уровень шумов
    # @param filterType - тип фильра (0 - Калмана; 1 - МНК)
    def __init__(self, object, trackID, deltaTime=0.25, accelNoiseMag=0.1, filterType=0):
        self.trackID = trackID
        self.trace = []
        self.trace.append(copy.deepcopy(object))
        self.lastObject = copy.deepcopy(object)
        self.predictionPoint = copy.deepcopy(object)
        self.skippedFrames = 0
        self.lastCorrectResult = copy.deepcopy(object)
        self.filterType = filterType
        if filterType == 0:
            self.__filter = filter.KalmanFilter6x6(self.predictionPoint, deltaTime, accelNoiseMag)
        else:
            self.__filter = filter.MNK(self.trace, copy.deepcopy(object))
        self.lastTimeDetect = 0



    ## @brief - функция расчета расстояния между последней и текущей точками
    # @param pt - текущая точка (TrackerObject)
    # @return - Вернет расстояние между точками
    def __CalcDist(self, pt):
        try:
            x = self.predictionPoint.point[0] - pt.point[0]
            y = self.predictionPoint.point[1] - pt.point[1]
            return math.sqrt((x ** 2) + (y ** 2))
        except Exception as e:
            print("Track:CalcDist", e)


    ## @brief - функция устанавливает значение deltaTime для фильтра калмана
    # @param dt - значение deltaTime
    # @return - Вернет расстояние между точками
    def setDeltaTime(self, dt):
        pass
        #if self.filterType == 0:
        #    self.__filter.resetDeltaTime(dt)


    ## @brief - функция обновления трека
    # @param object - новый объект (TrackerObject)
    # @param max_trace_length - максимальная длина трека
    def Update(self, object, max_trace_length):
        try:
            self.__PointUpdate(object)
            self.lastObject = copy.deepcopy(object)
            self.trace.append(copy.deepcopy(self.lastCorrectResult))

            if len(self.trace) > max_trace_length:
                self.trace.pop(len(self.trace) - max_trace_length - 1)
        except Exception as e:
            print("Track:Update", e)

    ## @brief - функция выполняет фильтрацию трека
    # @param pt - координаты точки
    def __PointUpdate(self, pt):
        try:
            dt = pt.timeDetect - self.trace[len(self.trace) - 1].timeDetect

            p = self.__filter.Update(pt)
            predict = self.__filter.GetPointPrediction(dt)

            self.lastCorrectResult = copy.deepcopy(p)
            self.predictionPoint = copy.deepcopy(predict)

        except Exception as e:
            print("Track:PointUpdate", e)


    ## @brief - функция запроса достоверности трека
    # @return - Вернет true or false
    @property
    def isTrue(self):
        return len(self.trace) <= self.skippedFrames + 1

    def __get_lin_regress(self, in_data, start_pos, in_data_size, xy):
        m1 = m2 = m3_x = m4_x = m3_y = m4_y = 0.0
        el_count = in_data_size - start_pos
        for i in range(start_pos, in_data_size):
            m1 += i
            m2 += i * i

            m3_x += in_data[i].point[0]
            m4_x += i * in_data[i].point[0]

            m3_y += in_data[i].point[1]
            m4_y += i * in_data[i].point[1]
        det_1 = 1 / (el_count * m2 - m1 * m1)
        m1 *= -1
        xy[0] = det_1 * (m1 * m3_x + el_count * m4_x)
        xy[1] = det_1 * (m2 * m3_x + m1 * m4_x)

        xy[2] = det_1 * (m1 * m3_y + el_count * m4_y)
        xy[3] = det_1 * (m2 * m3_y + m1 * m4_y)


if __name__ == '__main__':

    class TrackerObject:
        def __init__(self, x=0, y=0, vx=0, vy=0, intens=0, area=0, keypoint=None, timestamp=None):
            if timestamp is None:
                timestamp = time.time()
            self.point = (x, y)
            self.vx = vx
            self.vy = vy
            self.timeDetect = timestamp
            self.intens = intens
            self.area = area
            self.source = keypoint
    try:


        print(time.time())

        object = TrackerObject(10, 15)
        track = [Track(object, True)]
        dist = track[0].__CalcDist(TrackerObject(100, 150))

        track.append(Track(TrackerObject(120, 105), True, 0.1, 1))
        track[0].Update(TrackerObject(15, 20), True, 10)


    except Exception as e:
        print(e)