import ball_trees.spawn_sample_data as ssd
import ball_trees.generate_ball_tree as gbt
import ball_trees.knn as knn

b = gbt.Ball(0, ssd.labelled_data[:, :-1], ssd.labelled_data[:, -1], 20)

results = []
for i in range(ssd.unlabelled.shape[0]):
    results.append(knn.knn_ball_tree(3)(ssd.unlabelled[i, :], b))

print(results[:15])