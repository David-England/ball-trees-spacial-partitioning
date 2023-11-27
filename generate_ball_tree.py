import numpy as np


class Ball:
    def __init__(self, parent, data, max_elements):
        self.parent = parent
        self.data = data
        self.max_elements = max_elements

        if data.shape[0] > max_elements:
            self.__addChilds()
            self.is_leaf = False
        else:
            self.is_leaf = True

    def __furthest_from(self, point):
        furthest_index = 0
        furthest_dist = 0
        for i in range(0, self.data.shape[0]):
            _dist = dist(self.data[i, :], point)
            if _dist > furthest_dist:
                furthest_index = i
                furthest_dist = _dist
        return furthest_index

    def __dist_by_index(self, i, j):
        return dist(self.data[i, :], self.data[j, :])

    def __addChilds(self):
        centroid = np.array(self.data.mean(axis=0))
        anchor1_index = self.__furthest_from(centroid)
        anchor2_index = self.__furthest_from(self.data[anchor1_index, :])

        child1 = []
        child2 = []

        for i in range(0, self.data.shape[0]):
            if self.__dist_by_index(i, anchor1_index) > self.__dist_by_index(i, anchor2_index):
                child2.append(i)
            else:
                child1.append(i)

        self.Child1 = Ball(self, self.data[child1, :], self.max_elements)
        self.Child2 = Ball(self, self.data[child2, :], self.max_elements)


def dist(x, y):
    return np.sqrt(np.dot(y-x, y-x))