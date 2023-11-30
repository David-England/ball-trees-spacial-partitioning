import numpy as np
import generate_ball_tree as gbt


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
        labels = [p.label for p in self.points]
        label_values, label_counts = np.unique(labels, return_counts=True)
        # Institutes a bias in the multimodal case due to implicit sorting of keys.
        return label_values[np.argmax(label_counts)]


def knn_ball_tree(k):
    def _run_knn(k, point, ball, d_min_ancestors = 0, working_set = SortedPointSet()):
        d_min = max(gbt.dist(point, ball.centroid) - ball.radius, d_min_ancestors)
        if d_min >= working_set.d_sofar:
            return working_set

        # If not leaf, recurse on both children, starting with the closest (this optimisation not yet implemented).
        elif (not ball.is_leaf):
            working_set = _run_knn(k, point, ball.Child1, d_min, working_set)
            working_set = _run_knn(k, point, ball.Child2, d_min, working_set)

        else:
            for i in range(ball.data.shape[0]):
                d_point = gbt.dist(point, ball.data[i, :])
                if d_point < working_set.d_sofar:
                    working_set.add_point(ball.data[i, :], ball.labels[i], d_point)
                    while (len(working_set.points) > k):
                        working_set.bin_furthest_member()

        return working_set

    return lambda point, ball: _run_knn(k, point, ball).most_frequent_label()