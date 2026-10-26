# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

**Full paper:** 3D Gaussian Splatting for Real-Time Radiance Field Rendering

**Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis

**Published:** ACM Transactions on Graphics, Volume 42, Issue 4 (SIGGRAPH 2023)

**Project page:** [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

**Link:** [https://arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)

> **NOTE:** This is a first pass of the research paper. I feel that I lack some foundation in SfM, after which I plan to revisit this paper to understand everything again.

---

## 1. Gaussian Splats as Fancy Point Clouds

Gaussian splats are essentially just fancy point clouds.

In a point cloud, each point might contain:

$$
P_i = (\mathbf{x}_i, \mathbf{c}_i)
$$

where:

- $\mathbf{x}_i \in \mathbb{R}^3$ is its position.
- $\mathbf{c}_i \in \mathbb{R}^3$ is its color.

You project these points onto an image plane and render them.

The problem is that a mathematical point has zero area.

In contrast, the Gaussian splat analogue of a point is a **Gaussian primitive**.

Each Gaussian primitive contains:

$$
\mathcal{G}_i =
\left(
\boldsymbol{\mu}_i,
\boldsymbol{\Sigma}_i,
\alpha_i,
\mathbf{c}_i
\right)
$$

where:

- $\boldsymbol{\mu}_i$: 3D position (mean).
- $\boldsymbol{\Sigma}_i$: Covariance matrix (shape and orientation).
- $\alpha_i$: Opacity.
- $\mathbf{c}_i$: Color or view-dependent appearance, represented using spherical harmonics in the original method.

---

## 2. Initialization Using SfM

In fact, Gaussian splats are also literally trained with the first step being SfM.

Using COLMAP or some other library, we get back a sparse point cloud from the images along with camera extrinsics. Later on, over this point map, we initialize Gaussian splats.

The covariance is said to be a mixture of scale and orientation of the matrix. This is done as this allows for better covariance choices.

The scale is set to be equal to the mean of distances from nearest points, whereas orientation is set to $(1,0,0,0)$.

Opacity is also said to be a common value, and color is chosen as the color from the original point cloud.

---

## 3. Optimization and Density Control

Basically, we follow interleaved steps. Using SGD, we change the parameters of the Gaussians.

If the gradient for the position of a Gaussian is too large, then this could be due to two reasons.

### Overconstruction

A large Gaussian is present and is thus not able to define finer details. Here, we divide the Gaussian into two and shorten the scale of the daughter primitives.

### Underconstruction

In this case, what we do is add more Gaussians directly by creating a clone of the Gaussian. This clone is then moved to the new place.

---

## 4. Pruning and Opacity Resetting

Gaussians with $\alpha$ less than a threshold are removed. This also allows us to reduce the density of Gaussians.

Lastly, for floater cases, we regularly put $\alpha_i = 0$ for all $i$, then continue training. This allows for the removal of floaters further.