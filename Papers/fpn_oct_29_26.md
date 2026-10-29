# FPN

**Full paper:** Feature Pyramid Networks for Object Detection

**Authors:** Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie

**Published:** CVPR 2017

**Project page:** [Feature Pyramid Networks for Object Detection](https://ai.meta.com/research/publications/feature-pyramid-networks-for-object-detection/)

**Link:** [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

## Core Idea

Essentially, as we know, DETR faces issues when detecting small objects. This is partially due to using only the final feature map from the CNN backbone, which has a low spatial resolution and may lose important details about smaller objects. Older models used something like FPN to solve this issue. I am reading this paper to better understand the improvements made over the DETR model.

The main idea behind FPN is that a normal feedforward CNN creates multiple feature maps at different stages of the network. Each stage passes its output to the next stage for further processing and downsampling. What Feature Pyramid Networks do differently is that they also use the feature maps from the intermediate stages of the CNN. The deeper feature maps have stronger semantic information, whereas the earlier, higher-resolution feature maps preserve more spatial information. By combining both, they essentially bring out the best of both representations.

## Architecture

They use a CNN backbone, such as ResNet, and extract feature maps from different stages. The deeper feature maps are progressively upsampled, while the corresponding feature maps from earlier stages are passed through $1\times1$ convolutional layers to align their channel dimensions. Element-wise addition combines these representations at each stage, allowing the higher-resolution feature maps to incorporate semantic information from deeper layers. A subsequent $3\times3$ convolution produces the final feature maps at each pyramid level.

Thus, each resulting feature map contains both spatial details from its corresponding backbone stage and semantic information propagated from deeper stages. All these feature maps can then be used by the detector to make predictions at different scales.

**Note:** A feature pyramid does not output a single final feature map. Instead, it outputs a set of feature maps at different scales and resolutions.

## Results and Significance

FPN was evaluated with Faster R-CNN and other detection architectures, achieving state-of-the-art results at the time, including on the COCO benchmark. The key idea is to construct a hierarchy of semantically enriched feature maps at different resolutions, making it easier to detect objects of different sizes.