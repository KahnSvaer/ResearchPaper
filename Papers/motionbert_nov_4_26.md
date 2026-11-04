# November 4, 2026: MotionBERT

**Full paper:** MotionBERT: A Unified Perspective on Learning Human Motion Representations

**Authors**: Chuan Guo, Xinxin Zou, Sen Wang, Xinshuang Liu, Jian Li, and Yizhou Wang

**Published**: ICCV 2023

**Link**: [https://arxiv.org/abs/2210.06551](https://arxiv.org/abs/2210.06551)

## Architecture

Essentially, they have two streams that they join together based on the data using learned weights (W).

1. **Spatial Stream:** A spatial Transformer working independently on all joint positions within each frame. The goal is to understand the spatial relationships between different joints.
2. **Temporal Stream:** A temporal Transformer working on all joints throughout the video separately. It focuses on understanding how joint positions change across frames.

Both streams use standard Transformer architectures, just focused on different dimensions. They are later combined using learned attention weights to produce the final representation.

The entire thing operates on joint positions across the frames of a video.

## Pretraining

The main goal is to train an encoder that can be used to convert motion into a useful representation.

For the pretraining task, they went with 2D-to-3D lifting, with added masking for extra effect.

They used multiple data sources for training:

- **Large-scale mocap data:** Used with ground-truth 3D joint positions and a 3D reconstruction loss.
- **In-the-wild data:** Used with 2D observations and a reprojection loss.

The masking strategy forces the encoder to learn relationships between joints and across frames rather than simply relying on individual observations.

## Results

They were able to showcase strong pretrained results across multiple downstream tasks:

- 2D-to-3D pose lifting.
- Human mesh recovery.
- Action recognition.

Essentially, for testing this, they used lightweight task-specific decoders with a strong pretrained encoder. The encoder required only a relatively small amount of fine-tuning to perform well on downstream tasks.

## Takeaway

Honestly, the architecture itself is relatively simple. Both streams use standard Transformers, with the primary difference being whether they focus on spatial or temporal information.

The interesting part is how they combine masked motion modelling with multiple data sources to train a general-purpose motion encoder.

Rather than training separate models for every task, they demonstrate that a strong pretrained encoder can be reused across multiple human motion understanding tasks.