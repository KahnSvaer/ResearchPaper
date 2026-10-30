# October 30, 2026 - Deformable DETR

**Full paper:** Deformable DETR: Deformable Transformers for End-to-End Object Detection

**Authors:** Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai

**Published:** ICLR 2021

**Project page:** [https://huggingface.co/docs/transformers/en/model_doc/deformable_detr](https://huggingface.co/docs/transformers/en/model_doc/deformable_detr)

**Link:** [https://arxiv.org/abs/2010.04159](https://arxiv.org/abs/2010.04159)

**Extra Source Used:** [https://www.youtube.com/watch?v=9UG4amweIjk](https://www.youtube.com/watch?v=9UG4amweIjk)

This is an improvement over the DETR paper. I am reading this as a foundation of getting into world models (one of the foundation paper in reading list). This basically solves 2 problems faced by the DETR model.

1. DETR model seemed to be struggle with small objects
2. DETR model seemed to take a lot of time to converge with a lot of final attention maps being extremely sparse near the end of training.

## Intuitive understanding:

### Solution 1) Too long to converge (deformable attention)

For solving this problem the paper takes inspiration from deformable convolutions. Ie) Rather than multiplying for all values of K with q, we only look for specific components. This reduces the computation a lot making the convergence faster.

1. Rather than computing an attention matrix, the model creates attention weights directly as a linear projection of the query input. This significantly reduces the requirement of computation.
2. The model also reduce the Values computed for each head to some specific offsets rather than the entire sequence.
3. Just like the deformable convolutions these values are offsets from the original input. These offsets positions are also obtained through linear projection from the input embedding.
4. The computed offset could also be in decimals essentially requiring them to be a bilinear interpolation of the 4 grid points.

### Solution 2) Reduced ability to discover small objects (multiple feature maps)

1. Of course, since we have reduced the number of computation, we can increase the number of input features without issue. Thus we can use a set of feature map across different resolutions for better understanding of the image. We don’t use feature maps here as they empirically found it to not produce a significant improvement.
2. Across all the different feature map, they add a learnable embedding too for different resolutions feature maps. For the positional embedding they used standard sin cos embedding. Also the offsets are computed from the same feature map too.

---

## DETAILED IMPLEMENTATION

### 1. Deformable Attention

$$
\operatorname{DeformAttn}(z_q,p_q,x)
=
\sum_{m=1}^{M} W_m
\left[
\sum_{k=1}^{K}
A_{mqk}\, W'_m x\left(p_q+\Delta p_{mqk}\right)
\right]
$$

**Where:**

- $x \in \mathbb{R}^{C \times H \times W}$: Input feature map.
- $C$: Number of feature channels.
- $H,W$: Feature map height and width.
- $q$: Query index.
- $z_q \in \mathbb{R}^{C}$: Query content feature.
- $p_q \in \mathbb{R}^{2}$: 2D reference point.
- $M$: Number of attention heads.
- $m$: Attention-head index.
- $K$: Number of sampling points per head ($K \ll HW$).
- $k$: Sampling-point index.
- $\Delta p_{mqk} \in \mathbb{R}^{2}$: Learned sampling offset.
- $p_q+\Delta p_{mqk} \in \mathbb{R}^{2}$: Sampling coordinate.
- $A_{mqk} \in \mathbb{R}$: Attention weight, normalized with softmax over the $K$ sampling points for each head.
- $x(p_q+\Delta p_{mqk}) \in \mathbb{R}^{C}$: Feature vector sampled using bilinear interpolation.
- $W'_m \in \mathbb{R}^{d_h \times C}$: Per-head value projection matrix.
- $W_m \in \mathbb{R}^{C \times d_h}$: Per-head output projection matrix.
- $d_h=C/M$: Feature dimension per head.
- $\operatorname{DeformAttn}(\cdot) \in \mathbb{R}^{C}$: Final output feature.

For head $m$ and sampling point $k$, $W'_m x\left(p_q+\Delta p_{mqk}\right)$ is the equivalent projected value vector. Since only $K$ sampling points are used per head ($K \ll HW$), the computation processes far fewer values.

### 2. Linear projections of $z_q$ to predict the attention weights $A_{mqk}$ and sampling offsets $\Delta p_{mqk}$

Both sampling offsets and attention weights are predicted from the query feature $z_q$ using linear projections.

#### 1. Sampling offsets:

$$
\Delta P_q = W_{\mathrm{offset}}z_q+b_{\mathrm{offset}}
$$

The projection has $2MK$ outputs, which are reshaped into $M$ heads, $K$ sampling points, and 2D offsets:

$$
W_{\mathrm{offset}}\in\mathbb{R}^{2MK\times C},\qquad
b_{\mathrm{offset}}\in\mathbb{R}^{2MK},\qquad
\Delta P_q\in\mathbb{R}^{M\times K\times 2}
$$

Thus, each offset $\Delta p_{mqk}\in\mathbb{R}^{2}$.

#### 2. Attention weights:

$$
a_q=W_{\mathrm{attn}}z_q+b_{\mathrm{attn}}
$$

The projection produces $MK$ attention logits, then reshapes them into $M$ heads and $K$ sampling points:

$$
W_{\mathrm{attn}}\in\mathbb{R}^{MK\times C},\qquad
b_{\mathrm{attn}}\in\mathbb{R}^{MK},\qquad
a_q\in\mathbb{R}^{M\times K}
$$

Normalize the logits with a softmax over the $K$ sampling points within each head:

$$
A_{mqk}=
\frac{\exp(a_{mqk})}
{\sum_{j=1}^{K}\exp(a_{mqj})},
\qquad
A_q\in\mathbb{R}^{M\times K}
$$

#### Combined projection output:

Implementation wise, we train a concatenated form of the two projections matrix $W_{\mathrm{combined}}$ (with a combined bias):

$$
y_q=W_{\mathrm{combined}}z_q+b_{\mathrm{combined}},
\qquad
W_{\mathrm{combined}}\in\mathbb{R}^{3MK\times C},
\qquad
y_q\in\mathbb{R}^{3MK}
$$

This single projection produces $3MK$ outputs: split them into $2MK$ offset values (the $x$ and $y$ offset components for each of the $MK$ head–point pairs) and $MK$ attention logits, then reshape each to its corresponding tensor.

**Where:**

- $W_{\mathrm{offset}}$: Offset-prediction matrix; $b_{\mathrm{offset}}$: Offset-prediction bias.
- $W_{\mathrm{attn}}$: Attention-logit matrix; $b_{\mathrm{attn}}$: Attention-logit bias.
- $M$: Number of attention heads.
- $K$: Number of sampling points per head.
- $C$: Query embedding dimension.
- $\Delta p_{mqk}$: 2D sampling offset for head $m$, query $q$, point $k$.
- $a_{mqk}$: Pre-softmax attention logit; $A_{mqk}$: Normalized attention weight.

1. Using multi-scale features helps the model detect objects at different sizes. We take the ResNet feature maps $C_3$, $C_4$, and $C_5$, then apply a stride-2 $3\times3$ convolution to $C_5$ to produce the additional feature level $C_6$. We use L = 4, for implementation. Also we use a learned scale level embedding for dividing different resolution elements.
2. In encoder we replace all the blocks with this. In decoder we replace only the cross attention blocks this this new attention block keeping the original self attention block the same. Rest everything is same as DETR model.