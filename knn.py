import numpy as np
import generate_ball_tree as gbt


def knn_ball_tree(k):
    def _run_knn(k, point, ball, d_min_ancestors = 0, working_points = [], d_sofar = np.inf):
        d_min = max(gbt.dist(point, ball.centroid) - ball.radius, d_min_ancestors)
        if d_min >= d_sofar:
            return working_points, d_sofar

        # If not leaf, recurse on both children, starting with the closest (this optimisation not yet implemented).
        elif (not ball.is_leaf):
            working_points, d_sofar = _run_knn(k, point, ball.Child1, d_min, working_points, d_sofar)
            working_points, d_sofar = _run_knn(k, point, ball.Child2, d_min, working_points, d_sofar)

        else:
            for i in range(ball.data.shape[0]):
                d_point = gbt.dist(point, ball.data[i, :])
                if d_point < d_sofar:
                    working_points.append((ball.data[i, :], d_point, ball.labels[i]))
                    working_points.sort(key=lambda p: p[1])
                    while (len(working_points) > k):
                        working_points.pop()
                        d_sofar = working_points[-1][1]

        return working_points, d_sofar

    return lambda point, ball: _run_knn(k, point, ball)[0]