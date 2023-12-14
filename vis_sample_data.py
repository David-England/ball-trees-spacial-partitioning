from constructs import Ball
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
import matplotlib.patches as patches

def plot_ball_2D(ball: Ball, tree_depth: int):
    def _plonk_circle(axes, node: Ball, tree_depth: int, cmap_range: (int, int)):
        cmap_range_midpoint = int((cmap_range[0] + cmap_range[1])/2.)
        axes.add_patch(patches.Circle(node.centroid, node.radius, fill=False, ec=color_map(cmap_range[0])))
        if tree_depth > 1 and not node.is_leaf:
            _plonk_circle(axes, node.Child1, tree_depth-1, (cmap_range[0], cmap_range_midpoint))
            _plonk_circle(axes, node.Child2, tree_depth-1, (cmap_range_midpoint, cmap_range[1]))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(ball.data[:, 0], ball.data[:, 1], s=.25, c="k")
    _plonk_circle(ax, ball, tree_depth, cmap_range=(0, 255))

    fig.savefig('./sample_with_treetop.png')

color_map = cmaps["hsv"]