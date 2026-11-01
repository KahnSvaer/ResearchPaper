# PointNet++

**Full paper:** [https://arxiv.org/abs/1706.02413](https://arxiv.org/abs/1706.02413)

**Authors**: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas

**Published**: NIPS 2017

**Project page**: [https://stanford.edu/\~rqi/pointnet2/](https://stanford.edu/~rqi/pointnet2/)

**Link**: [https://arxiv.org/abs/1706.02413](https://arxiv.org/abs/1706.02413)

This paper is an extension of the point net paper. The old point net paper only ever completed the entire feature aggregation once globally, while this worked this didn’t really help much with understanding local subfeatures of the point cloud. So to combat this. the authors essentially created overlapping subpartitions over point clouds into subpoint clouds and added point net as local feature extractor.

Essentially they create an analogy that a pointnet could be viewed as a layer akin to CNN. Just like how CNN first has a small receptivity and later models have higher receptivity. We essentially apply point clouds to subpartitions. This reduces the number of points, then on the reduced subpoints we again partition and apply point clouds. With this we reach a final single group, apply point net over them to get better global features, which we use for classification. Just like CNNs point net thus act as a local feature extractor with increasing receptivity as the number of pointnet layers increase.

Essentially, a PointNet layer takes an input tensor of shape

$$
\mathbf{X} \in \mathbb{R}^{N' \times K \times (D + C)}
$$

where $N'$ is the number of sampled points, $K$ is the number of neighboring points per region, $D$ is the coordinate dimension (typically $D = 3$ for $(x,y,z)$), and $C$ is the number of input feature channels ($C = 0$ for the first layer).

It outputs a tensor of shape

$$
\mathbf{Y} \in \mathbb{R}^{N' \times (D + C')}
$$

where $C'$ is the number of output feature channels. ($N'$ is the number of sampled groups from the original $N$ points, so $N' < N$). Thus each pointnet layer decreases the number of points.

The partition of point clouds into overlapping subpartitions is done via FPS algorithm. Where we choose a random point as first centroid. The point farthest to point is the next centroid. The point with min farthest distance from chosen centroid is another centroid. We make this go on until we reach the number of centroids. Number of centroid as well as radius of grouping are taken as hyperparameters.

Now how to deal with sparse and unequal depth. The authors suggest 2 approaches.

### 1. MSG (Multi-scale grouping):

Here we just have multiple radius layers for our groups with same amount of centroids. We extract features from each group independently from same weighted pointnet. We just concatenate the features obtained from each radius for our final output. We also add an augmentation of random point drop to make the code more robust to sparsity.

**NOTE:** Each scale has different pointnet.

### 2. MRG (Multi-resolution grouping)

This above approach while works could be quite computationally expensive. So for the new approach for the later layers what they suggest is that apply only 2 point nets.

1. Apply point nets on the results obtained from previus point net.
2. Obtain another differently weighted point net on points from original point net (all the points lying within the chosen new radius of the later point cloud)
3. Concatenate the results

This is much more computationally efficient. The author mix and match both these variations with the original layer for ablations.

Now point net were also used for segmentation. So they go the reverse route they interpolate the features on points from 1 previous pointnet process ago and then also concatenate the local features from original. They then apply a 1x1 point net (Essentially sum relu layers added together) for reducing the number of channels. They go 1 layer back each time until we get the entire point cloud. Thus each point again have its own extracted feature and could be used for point labelling / part segmentation.