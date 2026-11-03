# November 3, 2026: MAE

**Full paper:** Masked Autoencoders Are Scalable Vision Learners

**Authors**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick

**Published**: CVPR 2022

**Project page**: https://facebookresearch.github.io/mae/

**Link**: https://arxiv.org/abs/2111.06377

Kinda a simple paper. In essence this is a generative model. They use an asymmetric encoder decoder architecture to train the encoder. They mask roughly 75% of the image patches and only pass unmasked vectors to the encoder (with PE). Later on they add a learned mask token to the output to represent patched token add PE and pass it to a decoder network which recreates the image. 

The decoder being smaller with the encoder only seeing small amount of image help increase the train speed of the model. Overall this is a cool early method of SSL on images. This can be used to train the encoder which can later on be used as a backbone for more complicated image generation tasks.