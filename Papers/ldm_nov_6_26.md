# November 6, 2026: LDM Model

**Full paper:** High-Resolution Image Synthesis with Latent Diffusion Models

**Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer

**Published:** CVPR 2022

**Project page:** https://ommer-lab.com/research/latent-diffusion-models/

**Link:** https://arxiv.org/abs/2112.10752

Before this work, diffusion models had already demonstrated promising results, but they required significant computational resources and training time. Previous papers had shown that diffusion models could outperform GANs and achieve state-of-the-art results.

The authors proposed training diffusion models in latent space instead of pixel space. They use an autoencoder to downsample the image into a 2D feature map and train the diffusion model over this latent representation.

They use a standard U-Net architecture for the diffusion process, with noise scheduling through DDPM/DDIM. They also introduced the integration of cross-attention layers into the U-Net for conditioning, allowing it to generate outputs based on inputs from a text encoder.

This work laid the foundation for Stable Diffusion, one of the major developments in image generation in 2022–2023.