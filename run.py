import datetime
import ball_trees.spawn_sample_data as ssd
from ball_trees.constructs import Ball
import ball_trees.knn as knn
import ball_trees.vis_sample_data as vsd

#---- Naive approach
dt_nv1 = datetime.datetime.now()

results_nv = []
for i in range(ssd.unlabelled.shape[0]):
    results_nv.append(knn.knn_naive(3)(ssd.unlabelled[i, :], ssd.labelled_data[:, :-1], ssd.labelled_data[:, -1]))

dt_nv2 = datetime.datetime.now()

#---- Ball tree approach
dt_bt1 = datetime.datetime.now()

b = Ball(0, ssd.labelled_data[:, :-1], ssd.labelled_data[:, -1], 20)

results_bt = []
for i in range(ssd.unlabelled.shape[0]):
    results_bt.append(knn.knn_ball_tree(3)(ssd.unlabelled[i, :], b))

dt_bt2 = datetime.datetime.now()

#---- Print results to console
print("NAIVE:", (dt_nv2 - dt_nv1).seconds, "s")
print(results_nv[:15])
print("BALL TREE:", (dt_bt2 - dt_bt1).seconds, "s")
print(results_bt[:15])

vsd.plot_ball_2D(b, 5)