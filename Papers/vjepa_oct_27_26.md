# **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video**

**Full paper:** Revisiting Feature Prediction for Learning Visual Representations from Video

**Authors:** Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas

**Published:** Preprint, arXiv, 2024

**Project page:** <https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/>

**Link:** <https://arxiv.org/abs/2404.08471>

---

V-JEPA extends the central idea of I-JEPA from images to videos. Instead of predicting the representations of hidden spatial image patches, it predicts the representations of hidden spatiotemporal regions.

1. **Tokenization:** I-JEPA operates on spatial image patches, whereas V-JEPA uses spatiotemporal tubelets/cuboids that combine temporal and spatial dimensions. Specifically, video is patchified across both spatial dimensions (a 16 × 16 patch in the reported implementation) and the temporal dimension (2 frames in the reported implementation). This makes the tokens resemble cuboids rather than the 2D rectangles used in image-based JEPA.

2. **Masking:** V-JEPA samples spatial target blocks and extends them across the entire temporal dimension of the video. It uses short-range and long-range masking configurations whose combined sampling produces an average masking ratio of approximately 90% in the reported configuration. This reduces the model's ability to exploit spatial and temporal redundancy in video to solve the prediction task trivially. Target regions span the full video duration, encouraging predictions that rely on context rather than only local spatiotemporal redundancy.

3. **Learning objective:** As in I-JEPA, a context encoder processes visible tokens, a predictor estimates the representations of masked regions, and an exponential moving average (EMA) target encoder supplies the target representations. V-JEPA uses an L1 feature-prediction loss rather than the L2 loss used in I-JEPA.

**Core idea:** Learn useful video representations by predicting hidden spatiotemporal features rather than reconstructing raw pixels.