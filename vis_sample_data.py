from constructs import Ball
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def plot_ball_2D(ball: Ball, tree_depth: int):
    def _plonk_circle(axes, node: Ball, tree_depth: int):
        axes.add_patch(patches.Circle(node.centroid, node.radius, fill=False))
        if tree_depth > 1 and not node.is_leaf:
            _plonk_circle(axes, node.Child1, tree_depth-1)
            _plonk_circle(axes, node.Child2, tree_depth-1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(ball.data[:, 0], ball.data[:, 1], s=.25)
    _plonk_circle(ax, ball, tree_depth)

    fig.savefig('./sample_with_treetop.png')