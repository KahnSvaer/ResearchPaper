# PointNet

**Full paper:** PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

**Authors:** Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas

**Published:** CVPR 2017

**Project page:** https://web.stanford.edu/~rqi/pointnet/

**Link:** https://arxiv.org/abs/1612.00593

So this is one of the first works on point cloud segmentation. Usually, works before this converted point clouds into more structured representations like voxels, which this paper does not do. Now, why is this hard? A point cloud follows permutation invariance, i.e., all the points in a point cloud are unordered. Further, if we rotate or translate the entire structure, it should still give the same results.

Since we are working with unordered points, we can't really combine their results directly. If we use RNNs, then we would be adding a forced sequence to them. Sequence is important here, and adding random augmentation won't really make the architecture invariant to permutations, so that approach is not ideal. Another approach could be some sort of sorting, but for high-dimensional point clouds, sorting is not really straightforward. The last option, which this paper ended up taking, is to use a symmetric function. Kinda like $+$ and $\times$. Both these functions are symmetric, and $g(a,b,c,\ldots)=g(b,c,a,\ldots)$ and all other permutations.

What PointNet does is use max pooling across dimensions (after increasing them) as its symmetric function. First, each point is independently passed through a shared MLP (with shared MLP processing ensuring permutation invariance; by shared, I mean that the same weighted MLP is applied to each point, not an RNN or sequence structure). Later, the features are aggregated to get a 1024-dimensional feature, which can be used by another MLP to predict the class.

Additionally, this 1024-dimensional feature can also be concatenated with the earlier 64-dimensional point features to perform point-wise segmentation.

Further, to add transformation robustness for the whole object, they also have a T-Net, a small MLP (similar architecture to PointNet, with independent processing of points using shared MLPs and aggregation using max pooling), to predict a transformation matrix. Though this can help in that regard, it does not guarantee invariance.

The orthogonality can be encouraged using an L2 regularizer:

$$
\mathcal{L}_{\mathrm{reg}} = \left\|I-TT^{\top}\right\|_F^2
$$

There are 2 T-Nets: one for the initial points and one after increasing the number of dimensions, which was shown empirically to increase accuracy. The orthogonality regularization is only for the later, higher-dimensional T-Net.

**NOTE:** Exact architecture is unimportant apart from these two ideas, tbh.