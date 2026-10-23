# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

Full paper: NeRF: Representing Scenes as Neural Radiance Fields for View
Synthesis

Authors: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan
T. Barron, Ravi Ramamoorthi, Ren Ng

Published: European Conference on Computer Vision (ECCV), 2020

Project page: <https://www.matthewtancik.com/nerf>

Link: <https://arxiv.org/abs/2003.08934>

## Base Idea / ABSTRACT

Essentially, NeRF trains an MLP to learn a continuous, implicit
representation of a 3D scene. Given spatial coordinates $(x, y, z)$ and
the viewing direction of a camera ray (represented by a 3D direction
vector or angular coordinates such as $\theta$ and $\phi$), the network
predicts the volume density ($\sigma$) and RGB color ($c$) at that
location.

The density depends on the spatial coordinates, while the color depends
on both the spatial coordinates and the viewing direction. This allows
NeRF to model view-dependent appearance, such as specular reflections.

To render an image from a novel viewpoint, NeRF samples points along
camera rays and integrates their predicted colors and densities using
differentiable volume rendering. By optimizing the network against
images captured from known camera viewpoints, it learns a representation
that can synthesize photorealistic images from previously unseen
viewpoints.

The original paper demonstrated state-of-the-art novel view synthesis
results on several benchmarks in 2020.

## WORKING

Essentially the MLP models are overtrained over the results or rather
each MLP works on an independent results. They don't just make the 3D
scene because the output color as well as take view direction as input
they are able to creates scenes with even added specularity/reflections
making it seem too real.

The 3D coordinate $\mathbf{x}$ is first passed through positional
encoding and then processed by an 8-layer MLP, which outputs the volume
density $\sigma(\mathbf{x})$ and a 256-dimensional feature vector
$\mathbf{h}(\mathbf{x})$.

**Then:**

The feature vector $\mathbf{h}(\mathbf{x})$ is concatenated with the
positional encoding of the viewing direction, $\gamma(\mathbf{d})$, and
passed through additional fully connected layers to predict the
view-dependent RGB color:

$$
[\mathbf{h}(\mathbf{x}), \gamma(\mathbf{d})]
\rightarrow \text{additional layers}
\rightarrow \mathbf{c}(\mathbf{x}, \mathbf{d})
$$

MLPs are trained by comparing colors. The ground-truth colors are
present in the images, and the predicted colors are calculated as
follows:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt
$$

where:

-   $T(t)$: transmittance (the fraction of light that survives to point
    $t$)
-   $\sigma(\mathbf{r}(t))$: density
-   $\mathbf{c}(\mathbf{r}(t),\mathbf{d})$: color

**Each infinitesimal segment contributes approximately:**

Contribution = Transmittance × Density × Color × Segment Length

This gives us the complete intuition:

-   **Transmittance** determines how much of the ray survives until the
    current point.
-   **Density** determines how strongly the local region contributes.
-   **Color** determines the color of that contribution.
-   **Integration** adds up contributions from the entire ray.

Here, transmittance is the exponential of the negative accumulated
density along the ray up to that point.

## Extra Tricks

**Positional Embedding** → They tested and found out that having a
higher dimension count at the start basically improved the results.
Essentially MLP tend to prefer focusing on lower frequency details
essentially, But my adding an extra positional embedding (aking to
transformer), we increase the starting values and this seem to make the
model focus better on higher frequency details like textures.

**Stratified Sampling** → Essentially taking the entire continuous
figures during test time is not really possible. The best that we could
do is essentially to take a numerical approximation over them using
quadrilature. Now quadrilature when done takes same exact points each
time, what they did in extra was literally made the sampled point more
random essentially. Divided the starting point and ending point over the
entire run of camera ray into N points. Each point is sampled uniformly
once from each of these buckets making the model learn about the entire
region over the course of the entire optimization, since each runtime
different points are chosen from each bucket with uniform probability.

**Hierarchical Sampling** → Now even with stratified sampling we are
getting multiple points that are completely useless parts that have
occluded completely or completely empty. So we do another thing
essentially. We simultaneously train 2 networks course network and fine
network. Essentially each run, the course network choose Nc samples to
optimize over. Later on based on their results Nf samples are found and
fine network optimize over Nc + Nf sample points essentially.

## Total Loss

$$
\mathcal{L} =
\sum_{\mathbf{r}\in\mathcal{R}}
\left[
\left\|\hat{C}_c(\mathbf{r})-C(\mathbf{r})\right\|_2^2
+
\left\|\hat{C}_f(\mathbf{r})-C(\mathbf{r})\right\|_2^2
\right]
$$

Where:

-   $\mathcal{R}$: Set of sampled camera rays.
-   $\hat{C}_c(\mathbf{r})$: Rendered color predicted by the coarse
    network.
-   $\hat{C}_f(\mathbf{r})$: Rendered color predicted by the fine
    network.
-   $C(\mathbf{r})$: Ground-truth pixel color.
-   $\|\cdot\|_2^2$: Squared $L_2$ error between predicted and
    ground-truth colors.

Thus both the fine MLP and course MLP are trained together.

## LIMITATIONS

1.  Interpretability - Producing the mesh directly allows for better
    usage as well as interpretability but encoding it within a MLP does
    not really do so.
2.  Optimization over the approach since this still takes in a large
    amount of time.
3.  Limitation with sparse observations
4.  Static scene assumption
5.  Slow rendering to new scene as each time all the points
    corresponding to the image are recalculated to view the scene.
