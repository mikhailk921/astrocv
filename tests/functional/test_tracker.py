import cv2
import numpy as np
import math
import astrocv.tracker.Tracker as Tracker
X = 0
Y = 0


def mv_mouseCallback(event, x, y, flags, param):
    global X, Y
    if event == cv2.EVENT_MOUSEMOVE:
        X = x
        Y = y


try:
    import random

    tracker = Tracker.Tracker(640, 480)
    frame = np.zeros((480, 640, 3))
    cv2.imshow("Video", frame)
    cv2.waitKey(1)

    MaxObjects = 4
    rndX = []
    rndY = []
    for i in range(0, MaxObjects):
        rndX.append(random.randint(-150, 150))
        rndY.append(random.randint(-100, 100))

    alpha = 0
    rotate = True
    intSqr = 10

    while True:
        alpha += 0.05
        frame = np.zeros((480, 640, 3))

        cv2.setMouseCallback("Video", mv_mouseCallback)
        o = []
        o.append(Tracker.TrackerObject(X, Y, 1000, 1000))
        # o.append(Tracker.TrackerObject(X+100, Y+100, 1000, 1000))
        # o.append(Tracker.TrackerObject(X+100, Y-100, 1000, 1000))
        # o.append(Tracker.TrackerObject(X-100, Y+100, 1000, 1000))
        # o.append(Tracker.TrackerObject(X-100, Y-100, 1000, 1000))


        shift = random.randint(0, MaxObjects - 1)
        for i in range(0, MaxObjects):
            shift += 1
            if shift >= MaxObjects:
                shift = 0
            if rotate:
                o.append(Tracker.TrackerObject(int(X + rndX[shift] * math.sin(alpha)), int(Y + rndY[shift] * math.cos(-alpha)),
                                       intSqr, intSqr))
            else:
                o.append(Tracker.TrackerObject(int(X + rndX[shift]), int(Y + rndY[shift]), intSqr, intSqr))

        tracker.Update(o)
        intSqr += 1
        try:
            # cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
            for i in range(0, len(tracker.track)):
                if len(tracker.track[i].trace) > 4:
                    for j in range(1, (len(tracker.track[i].trace) - 2)):
                        cv2.line(frame,
                                 (tracker.track[i].trace[j + 1].point[0], tracker.track[i].trace[j + 1].point[1]),
                                 (tracker.track[i].trace[j + 2].point[0], tracker.track[i].trace[j + 2].point[1]),
                                 (255, 0, 0), 3)
                    cv2.circle(frame,
                               (tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[0],
                                tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[1]),
                               5, (255, 0, 0), -1)
                    cv2.circle(frame,
                               (tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[0],
                                tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[1]),
                               2, (0, 0, 255), -1)

            for i in range(0, len(tracker.track)):
                st = None

                st = str(tracker.track[i].trackID) + " " + \
                     str(tracker.track[i].trace[len(tracker.track[i].trace) - 1].vx) + " " + \
                     str(tracker.track[i].trace[len(tracker.track[i].trace) - 1].vy)# + " " + \
                     #str(tracker.track[i].trace[len(tracker.track[i].trace) - 1].intens) + " " + \
                     #str(tracker.track[i].trace[len(tracker.track[i].trace) - 1].area)

                p = (tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[0] + 10,
                     tracker.track[i].trace[len(tracker.track[i].trace) - 1].point[1] - 10)
                cv2.putText(frame, st, p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            for i in range(0, len(o)):
                cv2.circle(frame, (o[i].point[0], o[i].point[1]), 3, (0, 0, 255), -1)

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