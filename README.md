# Ball tree spatial partitioning

## What's a ball tree?
A kind of binary tree, ball trees are a way to efficiently partition a dataset. They can be used to speed up kNN classification.

For every unlabelled data point requiring classification, a naive search computes the distance from this point to every labelled point. This is computationally expensive. Ball trees partition the dataset into a hierarchy of spheres; if, while computing the *k* nearest neighbours, the smallest distance from an unlabelled data point to a sphere is greater than the distance to a working *k*-th furthest point, we deduce that every point within the sphere is "too far" and can therefore be ignored. In this way, we can systematically eliminate large subsets of the dataset, potentially yielding a performance benefit.

To have a play, run the following scripts:
* *run.py*: runs a kNN search on synthetic two-dimensional data. We output two .png files: *sample.png* displays the labelled data, and *sample_with_treetop.png* overlays it with the top four levels of the ball tree (the jury is certainly out over whether it aids visual intuition or not). On my developer machine, the ball tree approach consistently yields a ~16x improvement in runtime over the naive search.
* *run_digits.py*: uses the algorithm to classify the well-known "digits" dataset with 98.9% accuracy. The literature suggests the execution time advantage of ball tree search disappears when the dimensionality of the data is high; this 64D example illustrates this.

## References
Wikipedia: https://en.wikipedia.org/wiki/Ball_tree

A couple of useful papers:
* For a description about how to generate a ball tree: https://arxiv.org/abs/1511.00628
* For an algorithm to perform kNN classification using a ball tree: https://www.jmlr.org/papers/volume7/liu06a/liu06a.pdf

See also k-d trees: https://en.wikipedia.org/wiki/K-d_tree