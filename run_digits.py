from datetime import datetime
import numpy as np
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from ball_trees.constructs import Ball
import ball_trees.knn as knn

data, labels = datasets.load_digits(return_X_y=True)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=.8, random_state=4)

#---- Naive approach
dt_nv1 = datetime.now()

results_nv = []
for i in range(data_test.shape[0]):
    results_nv.append(knn.knn_naive(3)(data_test[i, :], data_train, labels_train))

dt_nv2 = datetime.now()

#---- Ball tree approach
dt_bt1 = datetime.now()

b = Ball(None, data_train, labels_train, max_elements=20)

results_bt = []
for i in range(data_test.shape[0]):
    results_bt.append(knn.knn_ball_tree(3)(data_test[i, :], b))

dt_bt2 = datetime.now()

#---- Print results to console
new_section = lambda: print("\n--------")

new_section()
print("vs NAIVE:")
naive_vs_balltree = confusion_matrix(results_nv, results_bt)
print(naive_vs_balltree)
print(f"Agreement {naive_vs_balltree.trace() / naive_vs_balltree.sum() * 100. : .2f}%")

new_section()
print("vs CORRECT:")
label_vs_balltree = confusion_matrix(labels_test.tolist(), results_bt)
print(label_vs_balltree)
print(f"Accuracy {label_vs_balltree.trace() / label_vs_balltree.sum() * 100. : .2f}%")

new_section()
print("TIMING")
print("Naive approach:", (dt_nv2 - dt_nv1).microseconds, "us")
print("Ball tree approach:", (dt_bt2 - dt_bt1).microseconds, "us")