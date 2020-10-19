import math

## @brief - функция решения задачи назначения (жадный алгоритм)
# @param distMatrix - входной массив
# @param assign - массив назначений
# @param n - число треков
# @param m - число объектов
# @return - Вернет dict{"№ track": № object}
def solve(distMatrix, assign, n, m):
    try:
        passIndex = []
        if n <= m:
            for i in range(n):
                min = 999999999
                index = -1
                for j in range(m):
                    if passIndex.count(j) > 0:
                        continue
                    if distMatrix[i][j] < min:
                        min = distMatrix[i][j]
                        index = j
                assign[i] = index
                passIndex.append(index)
        else:
            for i in range(m):
                min = 999999999
                index = -1
                for j in range(n):
                    if passIndex.count(j) > 0:
                        continue
                    if distMatrix[j][i] < min:
                        min = distMatrix[j][i]
                        index = j
                assign[index] = i
                passIndex.append(index)
        res = {}
        for i in range(0, len(assign)):
            res[str(i)] = assign[i]
        return res
    except Exception as e:
        print("GreedyAlgorithm:Solve", e)

if __name__ == '__main__':

    import numpy as np

    a = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

    b = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

    print(a+b)
    t=3


    def assignmentProblemSolver(tracks, objects, assign):
        n = len(tracks)
        m = len(objects)

        distMatrix = [[.0 for i in range(0, m)]for j in range(0, n)]
        for i in range(0, n):
            for j in range(0, m):
                x = tracks[i][0] - objects[j][0]
                y = tracks[i][1] - objects[j][1]
                dist = math.sqrt(x * x + y * y)
                distMatrix[i][j] = dist
        solve(distMatrix, assign, n, m)

    try:
        point = {"x": 0, "y": 0}
        tracks = []
        tracks.append((400.0, 320.0))
        tracks.append((200.0, 320.0))
        tracks.append((100.0, 120.0))
        #tracks.append((300.0, 130.0))
        assignment = [-1 for i in range(0, len(tracks))]

        objects = []
        objects.append((384.0, 343.0))
        objects.append((284.0, 143.0))
        objects.append((184.0, 343.0))
        objects.append((84.0, 143.0))

        c = assignmentProblemSolver(tracks, objects, assignment)

        print(assignment)

        for i in range(0, len(assignment)):
            print("%d ->  assignment[ %d] = %d" % (assignment[i], i, assignment[i]))
            if assignment[i] != -1:
                pass

    except Exception as e:
        print(e)

