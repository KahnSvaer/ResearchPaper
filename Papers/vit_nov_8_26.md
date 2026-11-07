# November 8, 2026: ViT

**Full paper:** An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale

**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.

**Published:** ICLR 2021

**Project page:** [Google Research](https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/)

**Link:** [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

This paper is one of the earlier papers that bring the transformer architecture to images. The underlying idea is quite simple. Convert the original images into grids. Using an MLP, convert each grid into an embedding (MLP is learned while training). This MLP is known as a patch embedding. Like BERT, add a `[CLS]` token and add position embeddings over the grids. The paper uses learned positional embeddings as they showcased that learned embeddings didn't really improve and used extrapolation to add more embeddings in case the number of tokens is too much. Though later papers used standard sin-cos embeddings to avoid this extrapolation issue. Some RoPE embeddings have also recently come up that can be quite useful here.

Pass these tokens through a Transformer encoder consisting of multiple blocks of multi-head attention with layer norms and FFNs. The final results would be present at the position of the `[CLS]` token.

While initially they showed that ResNet performed better on smaller datasets due to translation equivariance inside ConvNets, Transformers show better scaling with large amounts of data, being able to be used for pretraining, as they have weaker inductive biases and thus need larger amounts of data to overcome that. Later on, for small datasets, they can just add a new classification head and go for linear probing for better results.