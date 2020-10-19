# -*- coding: utf-8 -*-
## @file tracker.py
# @brief Файл содержит класс трекера
import astrocv.tracker.Track as Track
import astrocv.tracker.HungarianAlg as HungarianAlg
import astrocv.tracker.GreedyAlgorithm as GreedyAlg
import math
import time

## @brief Класс объекта
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


## @brief Класс трекера
class Tracker:
    ## @brief - Конструктор класса
    # @param width - ширина
    # @param height - высота
    def __init__(self, width, height):

        self.settings = {
            "Width": width,
            "Height": height,
            "dt": 0.25,
            "accelNoiseMag": 0.1,
            "maxDistThres": 100,
            "maxSkippedFrames": 30,
            "maxTraceLength": 10,
            "methAssignProbSolv": 0,
            "methFiltering": 0,
        }
        self.track = []
        self.__nextTrackID = 0


    def __assignmentProblemSolver(self, tracks, objects, assign, method=0):
        n = len(tracks)
        m = len(objects)

        dMatrix = [[.0 for i in range(0, m)]for j in range(0, n)]
        distMatrix = [.0 for i in range(0, n * m)]
        for j in range(0, m):
            for i in range(0, n):
                x = tracks[i].predictionPoint.point[0] - objects[j].point[0]
                y = tracks[i].predictionPoint.point[1] - objects[j].point[1]
                dMatrix[i][j] = distMatrix[j * n + i] = math.sqrt(x * x + y * y)
        if method == 0:
            GreedyAlg.solve(dMatrix, assign, n, m)
        else:
            HungarianAlg.solve(distMatrix, assign, n, m)

        # удаление назначений пар с больщой дистанцией
        for i in range(0, len(assign)):
            if assign[i] != -1:
                if distMatrix[assign[i] * n + i] > self.settings["maxDistThres"]:
                    assign[i] = -1
                    self.track[i].skippedFrames += 1
            else:
                self.track[i].skippedFrames += 1


    ## @brief - функция обновления трекера
    # @param object - список текущих объектов (list(TrackerObject))
    def Update(self, objects):
        try:
            if type(objects) != type(list()):
                raise Exception("Tracker Exception: type(objects) != list {}".format(type(objects)))
            elif type(objects[0]) != type(TrackerObject()):
                raise Exception("Tracker Exception: type(objects[0]) != TrakerObjects {}".format(type(objects[0])))

        except Exception as e:
            print("Tracker:Update", e)

        try:
            assignment = [-1 for i in range(0, len(self.track))]

            #print("len track:", len(self.track))
            if len(self.track) > 0:
                # решение задачи назначения
                self.__assignmentProblemSolver(self.track, objects, assignment,
                                               method=self.settings["methAssignProbSolv"])

                # если трек не обнаружен долгое время, то удаляем
                while True:
                    if len(self.track) == 0:
                        break
                    j = 0
                    for i in range(0, len(self.track)):
                        j = i
                        if self.track[i].skippedFrames > self.settings["maxSkippedFrames"]:
                            self.track.pop(i)
                            assignment.pop(i)
                            break
                    if j == len(self.track) - 1:
                        break


            # поиск несоответствующих объектов и начало новых треков
            for i in range(0, len(objects)):
                if assignment.count(i) == 0:
                    self.track.append(Track.Track(objects[i],
                                                self.__nextTrackID,
                                                self.settings["dt"],
                                                self.settings["accelNoiseMag"],
                                                filterType=self.settings["methFiltering"]))
                    self.__nextTrackID += 1


            #  обновление трека
            for i in range(0, len(assignment)):
                if assignment[i] != -1:
                    self.track[i]._skippedFrames = 0
                    self.track[i].Update(objects[assignment[i]],
                                             self.settings["maxTraceLength"])

        except Exception as e:
            print("Tracker:Update", e)


if __name__ == '__main__':

    import cv2
    import numpy as np
    X = 0
    Y = 0

    def mv_mouseCallback(event, x, y, flags, param):
        global X, Y
        if event == cv2.EVENT_MOUSEMOVE:
            X = x
            Y = y

    try:
        import random

        tracker = Tracker(640, 480)
        tracker.settings["methAssignProbSolv"] = 1 #0 - greedy; 1 - hung
        tracker.settings["methFiltering"] = 1 #0 - kalman; 1 - MNK
        frame = np.zeros((480, 640, 3))
        cv2.imshow("Video", frame)
        cv2.waitKey(1)

        MaxObjects = 1
        rndX = []
        rndY = []
        for i in range(0, MaxObjects):
            rndX.append(random.randint(-150, 150))
            rndY.append(random.randint(-150, 150))

        alpha = 0
        rotate = False
        intSqr = 10

        while True:
            alpha += 0.05
            frame = np.zeros((480, 640, 3))

            cv2.setMouseCallback("Video", mv_mouseCallback)
            o = []
            #o.append(TrackerObject(X, Y, 1000, 1000))
            #o.append(TrackerObject(X+100, Y+100, 1000, 1000))
            #o.append(TrackerObject(X+100, Y-100, 1000, 1000))
            #o.append(TrackerObject(X-100, Y+100, 1000, 1000))
            #o.append(TrackerObject(X-100, Y-100, 1000, 1000))


            shift = random.randint(0, MaxObjects - 1)
            for i in range(0, MaxObjects):
                shift += 1
                if shift >= MaxObjects:
                    shift = 0
                if rotate:
                    o.append(TrackerObject(int(X + rndX[shift]*math.sin(alpha)), int(Y + rndY[shift]*math.cos(-alpha)), intSqr, intSqr,0 ,0))
                else:
                    o.append(TrackerObject(int(X + rndX[shift]), int(Y + rndY[shift]), intSqr, intSqr,0 ,0))

            tracker.Update(o)
            intSqr += 1
            try:
                #cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
                for i in range(0, len(tracker.track)):
                    if len(tracker.track[i].trace) > 4:
                        for j in range(1, (len(tracker.track[i].trace) - 2)):
                            cv2.line(frame, (int(tracker.track[i].trace[j+1].point[0]), int(tracker.track[i].trace[j+1].point[1])),
                                     (int(tracker.track[i].trace[j+2].point[0]), int(tracker.track[i].trace[j+2].point[1])),
                                     (255, 0, 0), 3)
                        cv2.circle(frame,
                                   (int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[0]),
                                    int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[1])),
                                   5, (255, 0, 0), -1)
                        cv2.circle(frame,
                                   (int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[0]),
                                    int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[1])),
                                   2, (0, 0, 255), -1)

                for i in range(0, len(tracker.track)):
                    st = None


                    st = str(tracker.track[i].trackID) + " " + \
                        str(int(tracker.track[i].trace[len(tracker.track[i].trace)-1].vx)) + " " + \
                        str(int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].vy)) + "\n" + \
                        str(int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].ax)) + " " + \
                        str(int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].ay))

                    p = (int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[0] + 10),
                        int(tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[1] - 10))
                    cv2.putText(frame, st, p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

                for i in range(0, len(o)):
                    cv2.circle(frame, (int(o[i].point[0]), int(o[i].point[1])), 3, (0, 0, 255), -1)

            except Exception as e:
                print("draw: ", e)


            cv2.imshow("Video", frame)
            val = cv2.waitKey(100)
            if val & 0xff == ord('q'):
                exit(0)
            if val & 0xff == ord('w'):
                continue


    except Exception as e:
        print(e)
