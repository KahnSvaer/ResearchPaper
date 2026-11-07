# November 7, 2026: RAFT

**Full paper:** RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

**Authors:** Zachary Teed, Jia Deng

**Published:** ECCV 2020

**Project page:** [https://princeton-vl.github.io/RAFT/](https://princeton-vl.github.io/RAFT/)

**Link:** [https://arxiv.org/abs/2003.12039](https://arxiv.org/abs/2003.12039)

This paper is basically about optical flow: estimating a dense 2D displacement between two frames, with one flow vector for each pixel. It is kinda a foundation paper for object tracking.

The pipeline is made up of these main blocks:

1. Context encoder and feature encoder
2. Visual similarity check
3. Iterative update using GRU-like networks

## Context encoder and feature encoder

The first step is feature extraction. This is done via a feature encoder and a context encoder (NOTE: context encoder is only for the initial frame, whereas the feature encoder works on any pair of frames including the initial.)

Note: Both of them have the same architecture essentially and are conv networks basically.

## Visual Similarity

Here the correlation of the features between the two images is found using dot products:

$C_{ijkl}=\sum_{d=1}^{D}G_1(i,j,d)G_2(k,l,d).$

Additionally, the correlation volume is average-pooled to create multiple smaller correlation volumes $C^1,C^2,C^3,C^4$. This is done only once per pair of frames.

## Iterative Step

Essentially we try to iteratively find the flow vector $\mathbf{f}_k(\mathbf{p})=(u_k(\mathbf{p}),v_k(\mathbf{p}))$ such that the matching point in the next frame is:

$\mathbf{q}_k(\mathbf{p})=\mathbf{p}+\mathbf{f}_k(\mathbf{p})=(x+u_k(\mathbf{p}),\;y+v_k(\mathbf{p})).$

This is iteratively found:

$\mathbf{f}_{k+1}(\mathbf{p})=\mathbf{f}_k(\mathbf{p})+\Delta\mathbf{f}_k(\mathbf{p}).$

We initialize $\mathbf{f}_0$ with 0 flow initially:

$\mathbf{f}_0(\mathbf{p})=\mathbf{0}.$

(In a video we can choose a warm initialization such that the previous pair's final flow is forward-projected to the next pair, with holes filled using nearest-neighbor interpolation.)

$\mathbf{f}_0^{\,t}=\mathcal{W}_{\mathrm{fwd}}\!\left(\hat{\mathbf{f}}_{K}^{\,t-1}\right).$

### Correlation Lookup

Now,

for each location in the source feature map we get one $\mathbf{q}_k$ position. The correlation for $x(i,j)$ that we take is $C(i,j,q_x,q_y)$, where $(q_x,q_y)=\mathbf{q}_k(\mathbf{p})$. At pyramid level $m$ the point is $\mathbf{q}_k(\mathbf{p})/2^m$. Around these positions we create a $(2r+1)\times(2r+1)$ neighborhood in the correlation volume and try to find the results. So the points become:

$\frac{\mathbf{q}_k(\mathbf{p})}{2^m}+(\Delta x,\Delta y)$

where $\Delta x,\Delta y$ belong to $\{-r,\ldots,r\}$.

Now if this point does not lie on an exact integer position, using bilinear interpolation on the surrounding 4 correlation values we find the scalar value. Assuming $r$ is 4, we get 81 positions per pyramid level. Since we have 4 levels, we get 324 values in the feature vector per point. This is the final correlation feature for that point.

The correlation feature of this step, flow features of the step as well as context encoder's direct results are concatenated together and passed as input to the GRU block for update, along with previous hidden state (for first step initialized with context encoder results with some conv layers to adjust the channels and for the later step we use the previous hidden state). The output of this block is $\Delta\mathbf{f}$. Then the flow is updated using $\mathbf{f}_{k+1}=\mathbf{f}_k+\Delta\mathbf{f}_k$. Thus we train the model.

## Loss

The loss that is taken into account is the sum of the L1 norm between each predicted flow $\mathbf{f}_i$ and the ground-truth flow $\mathbf{f}_{\mathrm{gt}}$:

$\mathcal{L}=\sum_{i=1}^{N}\gamma^{N-i}\left\|\mathbf{f}_{\mathrm{gt}}-\mathbf{f}_i\right\|_1,\qquad \gamma=0.8.$

Here, $\mathbf{f}_i$ is the flow prediction after update $i$, $\mathbf{f}_{\mathrm{gt}}$ is the ground-truth flow, and the $L_1$ norm is taken over the flow errors. The paper uses 12 update iterations during training. The paper supervises every flow prediction in the sequence, giving later predictions more weight to ensure that the model actually updates in an iterative manner.

## Upsampling

RAFT predicts optical flow at 1/8 resolution, so we need to upscale it to the original image resolution.

Instead of simply using bilinear interpolation, RAFT learns a convex upsampling mask using two convolutional layers.

For every coarse flow location, it predicts:

$8\times8\times9=576$

weights because each coarse location corresponds to 64 high-resolution pixels, and each of those pixels needs 9 weights.

Where does 9 come from? A $3\times3$ neighborhood of coarse flow vectors (8 surrounding vectors + the central one) around the position.

The weights go through softmax and are used to calculate a weighted combination of those 9 flow vectors for each high-resolution pixel. This allows us to get the weighted output of surrounding flows for all the 8x8 downsampled points.

Finally, we rearrange the resulting 64 vectors into the original $8\times8$ image region, allowing a creation of full scale resolution flow maps. This is only used for inferencing on the final f as well as loss function for training.