from constructs import Ball
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
import matplotlib.patches as patches

def plot_ball_2D(ball: Ball, tree_depth: int):
    def _plonk_circle(axes, node: Ball, tree_depth: int, color_sep: int):
        axes.add_patch(patches.Circle(node.centroid, node.radius, fill=False, ec=color_map(color_sep * (tree_depth-1))))
        if tree_depth > 1 and not node.is_leaf:
            _plonk_circle(axes, node.Child1, tree_depth-1, color_sep)
            _plonk_circle(axes, node.Child2, tree_depth-1, color_sep)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(ball.data[:, 0], ball.data[:, 1], s=.25, c="k")
    _plonk_circle(ax, ball, tree_depth, color_sep=int(255. / tree_depth))

    fig.savefig('./sample_with_treetop.png')

color_map = cmaps["hsv"]