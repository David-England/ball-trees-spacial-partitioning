import spawn_sample_data as ssd
import generate_ball_tree as gbt
import knn

b = gbt.Ball(0, ssd.labelled_data[:, :-1], ssd.labelled_data[:, -1], 20)

results = []
for i in range(ssd.unlabelled.shape[0]):
    results.append(knn.knn_ball_tree(3)(ssd.unlabelled[i, :], b))

print(results[:15])