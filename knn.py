import numpy as np
from ball_trees.constructs import dist


class LabelledPoint:
    def __init__(self, point, label, distance_to_point):
        self.point = point
        self.label = label
        self.distance_to_point = distance_to_point


class SortedPointSet:
    def __init__(self):
        self.points = []
        self.d_sofar = np.inf

    def add_point(self, new_point, new_label, distance_to_point):
        self.points.append(LabelledPoint(new_point, new_label, distance_to_point))
        self.points.sort(key=lambda p: p.distance_to_point)

    def bin_furthest_member(self):
        self.points.pop()
        self.d_sofar = self.points[-1].distance_to_point

    def most_frequent_label(self):
        labels = []
        counts = []

        for lab in [p.label for p in self.points]:
            if lab in labels:
                counts[labels.index(lab)] += 1
            else:
                labels.append(lab)
                counts.append(1)
        
        return labels[np.argmax(counts)]


def knn_ball_tree(k):
    def _run_knn(k, point, ball, d_min = 0, working_set = SortedPointSet()):
        if d_min >= working_set.d_sofar:
            return working_set

        elif (not ball.is_leaf):
            d_min_child1 = max(dist(point, ball.Child1.centroid) - ball.Child1.radius, d_min)
            d_min_child2 = max(dist(point, ball.Child2.centroid) - ball.Child2.radius, d_min)

            if (d_min_child1 <= d_min_child2):
                _run_knn(k, point, ball.Child1, d_min_child1, working_set)
                _run_knn(k, point, ball.Child2, d_min_child2, working_set)
            else:
                _run_knn(k, point, ball.Child2, d_min_child2, working_set)
                _run_knn(k, point, ball.Child1, d_min_child1, working_set)

        else:
            for i in range(ball.data.shape[0]):
                d_point = dist(point, ball.data[i, :])
                if d_point < working_set.d_sofar:
                    working_set.add_point(ball.data[i, :], ball.labels[i], d_point)
                    while (len(working_set.points) > k):
                        working_set.bin_furthest_member()

        return working_set

    return lambda point, ball: _run_knn(k, point, ball).most_frequent_label()


def knn_naive(k):
    def _run_knn(k, point, data, labels):
        working_set = SortedPointSet()

        for i in range(data.shape[0]):
            d_point = dist(point, data[i, :])
            if d_point < working_set.d_sofar:
                working_set.add_point(data[i, :], labels[i], d_point)
                while (len(working_set.points) > k):
                    working_set.bin_furthest_member()

        return working_set

    return lambda point, data, labels: _run_knn(k, point, data, labels).most_frequent_label()