# October 26, 2026 - IJEPA

**Full paper:** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

**Authors:** Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Nicolas Ballas, Florian Strub, Alexei Baevski, Stanisław Zaięba, et al.

**Published:** CVPR 2023

**Project page:** [https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)

**Link:** [https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243)

---

Taking inspiration from masked image modelling approaches, the idea is that we learn to predict the embedding of a target block using some context block, rather than the actual pixel wise patch of the target block. This helps as this allows the encoders to focus more on the semantic meaning behind the patch inplace of high level features like fur, texture color etc. Allowing for a much better and higher semantic meaning encoding.

There are in total 3 models. Target encoder, context encoder and lastly predictor (all made with VIT blocks for the paper implementation). We essentially only train context encoder and the predictor during runtime whereas the target encoder is simply the ema of the context encoder. This is how the data flows:

1. We first divide the entire image into smaller parts.
2. We pass the entire image all of the patches in the target encoder. Suppose if we had divided the image into N parts, then we would get sequence of N vectors essentially(that is how we actually divide the image).
3. Out of this we sample M blocks all of which are target (repetition allowed while sampling target blocks). We take large target blocks largely square (or at least rectangle form). Large target blocks encourage larger semantic ambiguity in SSL context.
4. For the remaining we are essentially left with context block after removing the used up blocks essentially.
5. This context blocks are passed through context encoder alone (without passing in the target information). This gives us context embedding.
6. Over this context embedding we add learnable masked tokens along with positional sin cos encoding to give information about where the patches come from.
7. We pass this context embedding along with PE and masked token and get a new embedding essentially. We compare the L2 loss of the sampled target block as well as corresponding vector position from the predictors embedding.

NOW, there is an important point here if we are only comparing the prediction can’t there be a degenerate solution present in the form of always predicting constant values. This can technically happen but the asymmetric nature between the target encoder and context encoder makes it less likely.