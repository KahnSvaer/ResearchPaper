# November 10, 2026: YOLO

**Full paper:** You Only Look Once: Unified, Real-Time Object Detection

**Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

**Published:** CVPR 2016

**Project page:** https://pjreddie.com/darknet/yolo/

**Link:** https://arxiv.org/abs/1506.02640

One of the first paper working end to end for real time object detection by framing it as single regression problem. They devide the image into SxS regions. If the bounding box center lies within that region, it gets assigned to this region. Each region outputs (B*5+C) output, where C = class prob (=number of classes) and 5 is essetnially for x,y,h,w. The x,y tells where the center of the bounding box is relative to the grid and h,w tells us the height and wifth of the bounding box compared to the image. There is also a confidence score predicted per box based purely on if it detects an object there. The final text confidence is represented by: P(class_i | Object) * P(object), Also non max suppression is used to increase correctness to ensure that there is only a single bounding box per object. The bounding box loss is given precedence over the confidence scores to reduce model instability due to no class options. Overall square sum loss (L2 loss) is used throughout with a linear layer as the final output (even for class prob somehow). Also to improve results over small box predictions too, square roots of w and h also predicted out. Also an interesting point to note was that even at this point in convnet history, concept of pretraining was still progressing as they use large imagenet dataset to pretrain it and later convert it to object detection problem. One thing that i also found cool is the interesting project page lol. Recent pages look way too sanitized over the chaotic page presented here.