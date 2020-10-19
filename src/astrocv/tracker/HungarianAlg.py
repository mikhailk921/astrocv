# -*- coding: utf-8 -*-
## @file HungarianAlg.py
# @brief Файл содержит реализацию венгерского алгоритма
import math
## @brief - функция решения задачи назначения
# @param distMatrixIn - входной массив
# @param assign - массив назначений
# @param N - число треков
# @param M - число объектов
# @return - Вернет cost
def solve(distMatrixIn, assign, N, M):
    try:
        cost = [0]
        __assignmentoptimal(assign, cost, distMatrixIn, N, M)
        return cost
    except Exception as e:
        print("AssignmentProblemSolver:Solve", e)

## @brief - функция реализации оптимальных назначений
# @param assignment - массив назначений
# @param cost - цена
# @param distMatrixIn - входной массив
# @param N - число треков
# @param M - число объектов
def __assignmentoptimal(assignment, cost, distMatrixIn, N, M):
    nOfRows = N
    nOfColumns = M
    #print(N, M)
    #print(distMatrixIn)
    nOfElements = nOfRows * nOfColumns
    distMatrix = [0 for i in range(0, nOfElements)]
    distMatrixEnd = len(distMatrix)

    for row in range(0, nOfElements):
        value = distMatrixIn[row]
        assert (value >= 0)
        distMatrix[row] = value

    coveredColumns = [False for i in range(0, nOfColumns)]
    coveredRows = [False for i in range(0, nOfRows)]
    starMatrix = [False for i in range(0, nOfElements)]
    primeMatrix = [False for i in range(0, nOfElements)]
    newStarMatrix = [False for i in range(0, nOfElements)]

    if nOfRows <= nOfColumns:
        for row in range(0, nOfRows):
            Temp = row
            minValue = distMatrix[Temp]
            Temp += nOfRows

            while Temp < distMatrixEnd:
                value = distMatrix[Temp]
                if value < minValue:
                    minValue = value
                Temp += nOfRows

            Temp = row
            while Temp < distMatrixEnd:
                distMatrix[Temp] -= minValue
                Temp += nOfRows

        # step 1 and 2a
        for row in range(0, nOfRows):
            for col in range(0, nOfColumns):
                if distMatrix[nOfRows * col + row] == 0:
                    if coveredColumns[col] is False:
                        starMatrix[nOfRows * col + row] = True
                        coveredColumns[col] = True
                        break

    # if nOfRows > nOfColumns
    else:
        for col in range(0, nOfColumns):
            Temp = nOfRows * col
            columnEnd = Temp + nOfRows
            minValue = distMatrix[Temp]
            Temp += 1
            while Temp < nOfRows:
                value = distMatrix[Temp]
                Temp += 1
                if value < minValue:
                    minValue = value

            Temp = nOfRows * col
            while Temp < nOfRows:
                distMatrix[Temp] -= minValue
                Temp += 1

        # step 1 and 2a
        for col in range(0, nOfColumns):
            for row in range(0, nOfRows):
                if distMatrix[row + nOfRows * col] == 0:
                    if coveredRows[row] is False:
                        starMatrix[row + nOfRows * col] = True
                        coveredColumns[col] = True
                        coveredRows[row] = True
                        break

        for row in range(0, nOfRows):
            coveredRows[row] = False

    minDim = 0
    if nOfRows <= nOfColumns:
        minDim = nOfRows
    else:
        minDim = nOfColumns

    # move to step 2b
    __step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
                nOfRows, nOfColumns, minDim)

    # compute cost and remove invalid assignments
    __computeassignmentcost(assignment, cost, distMatrixIn, nOfRows)

## @brief - функция реализующая один из шагов алгоритма
def __step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
            nOfRows, nOfColumns, minDim):
    # count covered columns
    nOfCoveredColumns = 0
    for col in range(0, nOfColumns):
        if coveredColumns[col] is True:
            nOfCoveredColumns += 1

    if nOfCoveredColumns == minDim:
        # algorithm finished
        __buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns)
    else:
        # move step 3
        __step3_5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
                        nOfRows, nOfColumns, minDim)

## @brief - функция собирает вектор назначений
def __buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns):
    for row in range(0, nOfRows):
        for col in range(0, nOfColumns):
            if starMatrix[row + nOfRows * col]:
                assignment[row] = col
                break

## @brief - функция высчитывает стоимость назначений
def __computeassignmentcost(assignment, cost, distMatrixIn, nOfRows):
    for row in range(0, nOfRows):
        col = assignment[row]
        if col >= 0:
            cost[0] += distMatrixIn[row + nOfRows * col]

## @brief - функция одного из шагов алгоритма
def __step3_5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
            nOfRows, nOfColumns, minDim):
    while True:
        # step 3
        zerosFound = True
        while zerosFound:
            zerosFound = False
            for col in range(0, nOfColumns):
                if coveredColumns[col] is False:
                    for row in range(0, nOfRows):
                        if ((coveredRows[row] is False) and (distMatrix[row + nOfRows * col] == 0)):
                            # prime zero
                            primeMatrix[row + nOfRows * col] = True

                            # find starred zero in current row
                            starCol = 0
                            for i in range(0, nOfColumns):
                                starCol = i
                                if starMatrix[row + nOfRows * i] is True:
                                    break
                                else:
                                    if i == nOfColumns - 1:
                                        starCol += 1

                            if starCol == nOfColumns:  # no starred zero found
                                # move to step 4
                                __step4(assignment, distMatrix, starMatrix, newStarMatrix,
                                            primeMatrix, coveredColumns, coveredRows, nOfRows,
                                            nOfColumns, minDim, row, col)
                                return
                            else:
                                coveredRows[row] = True
                                coveredColumns[starCol] = False
                                zerosFound = True
                                break

        # step 5
        h = 9999999999999
        for row in range(0, nOfRows):
            if coveredRows[row] is False:
                for col in range(0, nOfColumns):
                    if coveredColumns[col] is False:
                        value = distMatrix[row + nOfRows * col]
                        if value < h:
                            h = value

        # add h to each covered row
        for row in range(0, nOfRows):
            if coveredRows[row] is True:
                for col in range(0, nOfColumns):
                    distMatrix[row + nOfRows * col] += h

        # subtract h from each uncovered column
        for col in range(0, nOfColumns):
            if coveredColumns[col] is False:
                for row in range(0, nOfRows):
                    distMatrix[row + nOfRows * col] -= h

## @brief - функция одного из шагов алгоритма
def __step4(assignment, distMatrix, starMatrix, newStarMatrix,
            primeMatrix, coveredColumns, coveredRows, nOfRows,
            nOfColumns, minDim, row, col):
    nOfElements = nOfRows * nOfColumns

    # generate temporary copy of starMatrix
    for n in range(0, nOfElements):
        newStarMatrix[n] = starMatrix[n]

    # star current zero
    newStarMatrix[row + nOfRows * col] = True

    # find starred zero in current column
    starCol = col
    starRow = 0

    for i in range(0, nOfRows):
        starRow = i
        if starMatrix[i + nOfRows * starCol] is True:
            break
        else:
            if i == nOfRows - 1:
                starRow += 1


    while starRow < nOfRows:
        # unstar the starred zero
        newStarMatrix[starRow + nOfRows * starCol] = False

        #find primed zero in current row
        primeRow = starRow
        primeCol = 0
        for i in range(0, nOfColumns):
            primeCol = i
            if primeMatrix[primeRow + nOfRows * primeCol] is True:
                break
            else:
                if i == nOfColumns - 1:
                    primeCol += 1

        #star the primed zero
        newStarMatrix[primeRow + nOfRows*primeCol] = True

        #find starred zero in current column
        starCol = primeCol
        for i in range(0, nOfRows):
            starRow = i
            if starMatrix[i + nOfRows * starCol] is True:
                break
            else:
                if i == nOfRows - 1:
                    starRow += 1

    # use temporary copy as new starMatrix
    # delete all primes, uncover all rows
    for n in range(0, nOfElements):
        primeMatrix[n] = False
        starMatrix[n] = newStarMatrix[n]
    for n in range(0, nOfRows):
        coveredRows[n] = False

    # move to step 2a
    __step2a(assignment, distMatrix, starMatrix, newStarMatrix,
                primeMatrix, coveredColumns, coveredRows, nOfRows,
                nOfColumns, minDim)

## @brief - функция одного из шагов алгоритма
def __step2a(assignment, distMatrix, starMatrix, newStarMatrix,
                    primeMatrix, coveredColumns, coveredRows, nOfRows,
                    nOfColumns, minDim):

    for col in range(0, nOfColumns):
        startMatrixTemp = nOfRows * col
        columnEnd = startMatrixTemp + nOfRows

        while startMatrixTemp < columnEnd:
            if starMatrix[startMatrixTemp] is True:

                coveredColumns[col] = True
                break
            startMatrixTemp += 1

    __step2b(assignment, distMatrix, starMatrix, newStarMatrix,
                primeMatrix, coveredColumns, coveredRows, nOfRows,
                nOfColumns, minDim)


if __name__ == '__main__':

    try:

        point = {"x": 0, "y": 0}
        tracks = []
        '''tracks.append((400.0, 320.0))
        tracks.append((200.0, 320.0))
        tracks.append((100.0, 120.0))
        tracks.append((300.0, 130.0))

        objects = []
        objects.append((384.0, 343.0))
        #objects.append((284.0, 143.0))
        objects.append((184.0, 343.0))
        #objects.append((84.0, 143.0))'''

        tracks.append({"x": 400.0, "y": 320.0})
        tracks.append({"x": 200.0, "y": 320.0})
        tracks.append({"x": 100.0, "y": 120.0})
        #tracks.append({"x": 300.0, "y": 130.0})

        objects = []
        objects.append({"x": 384.0, "y": 343.0})
        objects.append({"x": 284.0, "y": 143.0})
        objects.append({"x": 184.0, "y": 343.0})
        objects.append({"x": 84.0, "y": 143.0})

        '''tracks.append({"x": 399.0, "y": 323.0})
        tracks.append({"x": 399.0, "y": 123.0})
        tracks.append({"x": 199.0, "y": 323.0})
        tracks.append({"x": 199.0, "y": 123.0})

        objects = []
        objects.append({"x": 284.0, "y": 443.0})
        objects.append({"x": 284.0, "y": 243.0})
        objects.append({"x": 84.0, "y": 443.0})
        #objects.append({"x": 84.0, "y": 243.0})'''

        N = len(tracks)
        M = len(objects)
        assignment = [-1 for i in range(0, N)]
        Cost = [0 for i in range(0, M * N)]

        maxCost = 0
        for i in range(0, len(tracks)):
            tmp = []
            for j in range(0, len(objects)):
                point["x"] = tracks[i]["x"] - objects[j]["x"]
                point["y"] = tracks[i]["y"] - objects[j]["y"]
                dist = math.sqrt(point["x"] ** 2 + point["y"] ** 2)
                Cost[i + j * N] = dist
                tmp.append(dist)
                if dist > maxCost:
                    maxCost = dist
            print(tmp)

        #APS = AssignmentProblemSolver()
        c = solve(Cost, assignment, N, M)

        print(assignment)

        for i in range(0, len(assignment)):
            print("Cost[%d] = %d ->  assignment[ %d] = %d" % (assignment[i] * N,
                                                              Cost[i + assignment[i] * N], i, assignment[i]))
            if assignment[i] != -1:
                pass

    except Exception as e:
        print(e)
