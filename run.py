import ball_trees.spawn_sample_data as ssd
from ball_trees.constructs import Ball
import ball_trees.knn as knn

b = Ball(0, ssd.labelled_data[:, :-1], ssd.labelled_data[:, -1], 20)

#---- Naive approach
results_nv = []
for i in range(ssd.unlabelled.shape[0]):
    results_nv.append(knn.knn_naive(3)(ssd.unlabelled[i, :], ssd.labelled_data[:, :-1], ssd.labelled_data[:, -1]))

#---- Ball tree approach
results_bt = []
for i in range(ssd.unlabelled.shape[0]):
    results_bt.append(knn.knn_ball_tree(3)(ssd.unlabelled[i, :], b))

#---- Print results to console
print(results_nv[:15])
print(results_bt[:15])