import numpy as np
from matplotlib import pyplot as plt

np.random.seed(4)
rng = np.random.default_rng(seed=4)

x1 = [4., 12.] + [1., 1.5] * np.random.randn(4096, 2)
x2 = [3., 9.] + [2.5, .8] * np.random.randn(2048, 2)
unlabelled = [5., 10.] + [1.5, 1.] * np.random.randn(8192, 2)

x1 = np.concatenate((x1, np.zeros((4096, 1)) + 1.), axis=1)
x2 = np.concatenate((x2, np.zeros((2048, 1)) + 2.), axis=1)
labelled_data = np.concatenate((x1, x2), axis=0)
rng.shuffle(labelled_data)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x1[:, 0], x1[:, 1], s=.25, c='b')
ax.scatter(x2[:, 0], x2[:, 1], s=.25, c='r')
fig.savefig('./sample.png')