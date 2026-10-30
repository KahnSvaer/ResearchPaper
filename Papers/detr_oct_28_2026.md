# End-to-End Object Detection with Transformers

**Full paper:** End-to-End Object Detection with Transformers

**Authors:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko

**Published:** ECCV 2020

**Project page:** [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)

**Link:** [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)

The authors reframe object detection task as a global set prediction problem. Unlike other methods they don’t require hand crafted features like anchors or post processing steps like non max supression.

## ARCHITECTURE

### 1. Backbone

A CNN backbone converts image to a feature map first.

### 2. Transformer Encoder

Later a 1x1 CNN block change this to a feature map with smaller sized channel. This new feature map is flattened to convert it to a sequence before passing it to a transformer encoder. A fixed PE is added to the input of each self-attention layer.

### 3. Transformer Decoder

We pass N learnable vectors as input, AKA query vectors. We pass the result from encoder as K and V (cross attention). While we do multiple passes while adding the positional embedding (query vectors) each time, though all the reult are processed parallely.

### 4. Prediction Feed-Forward Networks (FFNs)

The final prediction is computed by a 3-layer perceptron with ReLU activation function and hidden dimension d, and a linear projection layer. The FFN predicts the normalized center coordinates, height and width of the box w.r.t. the input image, i.e. $(c_x, c_y, w, h)$, and the linear layer predicts the class label using a softmax function. An additional special class label $\varnothing$ is used to represent that no object is detected within a slot. So essenitally 2 seperate heads on each of the N vectors obtained from the transformer.

## LOSS

The results that we obtain from the DETR model are in the form of unordered sets. Thus, before comparing the results, we need to match elements and class labels to the ground truth. This is done by finding the minimum loss using Hungarian matching. Each element of the ground-truth label can be represented as $(c_i, b_i)$, where $c_i$ is the target class label (which may be $\varnothing$), and $b_i \in [0, 1]^4$ is a vector that defines the ground-truth box's center coordinates, height, and width relative to the image size.

$$ \mathcal{L}_{\text{match}}(y_i,\hat{y}_{\sigma(i)}) = -\mathbb{1}_{c_i \neq \varnothing}\hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing}\mathcal{L}_{\text{box}}(b_i,\hat{b}_{\sigma(i)}) $$

The Lbox is found via:

$$\mathcal{L}_{\text{box}}(b_i,\hat{b}_{\sigma(i)})=\lambda_{\mathrm{iou}}\mathcal{L}_{\mathrm{iou}}(b_i,\hat{b}_{\sigma(i)})+\lambda_{L_1}\|b_i-\hat{b}_{\sigma(i)}\|_1$$

$$
\text{where } \lambda_{\mathrm{iou}}, \lambda_{L_1} \in \mathbb{R} \text{ are hyperparameters.}
$$

## LIMITATIONS

1. **Slow convergence:** DETR requires substantially more training epochs than many conventional object detectors to achieve strong performance.
2. **Weak performance on small objects:** DETR performs particularly well on large objects but struggles with smaller objects. The relatively coarse feature representation and the difficulty of learning object queries contribute to this limitation.
3. **Computational cost:** Global self-attention over spatial image features can be expensive at high resolutions. Although DETR achieves competitive inference speed against Faster R-CNN, its architecture is not specifically optimized for maximum real-time throughput.