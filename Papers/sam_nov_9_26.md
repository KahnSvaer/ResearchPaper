# November 9, 2026: SAM

**Full paper:** Segment Anything

**Authors:** Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick

**Published:** ICCV 2023

**Project page:** [https://aidemos.meta.com/segment-anything/](https://aidemos.meta.com/segment-anything/)

**Link:** [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

Sam has 2 main contributions:

1. A foundation model for Segmentation
2. A large dataset to train segmentation foundation models

## Foundation Model

Foundation model has 3 main parts:

1. Image Encoder
2. Prompt Encoder
3. Mask Decoder

#### Image Encoder

They use a MAE pretrained ViT-H/16 model minimally adapted for high res inputs. It runs once per image.

#### Prompt Encoder

The prompts encoder can take 2 types of inputs - Sparse (Bounding Box, Point) and Dense (Mask)

Sparse prompts are mapped to 256D vectorial embeddings as follows.

- A point is represented as the sum of a positional encoding of the point’s location and one of two learned embeddings that indicate if the point is either in the foreground or background.
- A box is represented by an embedding pair of top left and bottom right corner. The positional encoding of corner summed with a learned embedding representing “top-left corner” or “bottom-right corner”
- For text they use the text encoder from CLIP
- NOTE: Sparse prompt could be a zero embedding vector if only dense prompt ie masks are provided.

Dense prompts are rough masks that can also be used to refine the results. They downsample the mask convert it to embedding dimensions and add it element wise to the image embedding. If no mask present a learnable embedding of “no mask” is added.

#### Mask Encoder

For the mask encoder they take in information from both prompt encoder and image encoder and create a mask for both of them. They extensively use ouputs of both results such that prompt initially act as query for stage 1 and for stage 2 act as keys. Positional embedding are added at each step and a 2way transformer block is used.

With this they are able to output multiple MASK blocks for each position.

The model outputs 3 mask tokens to account for ambiguity in prompts (thus, accounting for nested masks). They are ranked by seperate confidence scores outputted by the model. The model is also trained such that for each image embedding is calculated once and different prompts are then used to find out the masks over the image.

## Dataset

They contributed a large 1B+ mask with 11M+ images. The images were licensed from a photo provider with private information like number plates as well as faces removed,

They followed a model in the loop approach for creating the mask. This was done in 3 steps.

1. assisted-manual - Model creates initial masks and human correct the masks manually
2. semi-automatic - Mostly done by the model later humans add in further masks
3. fully automatic - Done entirely by the model

They tested out the results across social and qualitative factors. Focusing here on the qualitative factors they sampled 500 images. On this they did manually annotation and compared the results. They found out that 94% of the images had an IOU over 90% showing strong results.

One suprising thing is that the SAM results showed great results on quite a number of other zero shot tasks like instance segmentation, text to mask, object proposal and edge detections.