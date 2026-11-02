# VoxelNet

**Full paper:** VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

**Authors:** Yin Zhou, Oncel Tuzel

**Published:** CVPR 2018

**Project page:** [https://machinelearning.apple.com/research/voxelnet-end-to-end](https://machinelearning.apple.com/research/voxelnet-end-to-end)

**Link:** [https://arxiv.org/abs/1711.06396](https://arxiv.org/abs/1711.06396)

This is a voxel-based method of point cloud processing. Previous voxel-based methods used handcrafted features per voxel, based on statistics such as mean, point distribution and density, which introduced reliance on manually designed features. Further lidar based data can contain upward of 100K+ points so point net like architectures are also not really possible. What they did was essentially divided the entire point cloud into subsections of voxels.

Each point in a voxel was grouped together and using a pointnet like network converted to feature encoding. (Not the same architecture but with the same base principles in essence). 3DConv layers are used to aggregate the voxel features together (I believe 0 vector was used for empty voxels). By representing voxels as sparse vector they reduce the computational complexity, and by using conv3D they are able to understand local structure better too.

Later on on the global encoding, they apply a region based proposal network for 3D object detection focusing on center position(x,y,z), dims(l,b,h) and yaw(theta). The loss function is a variant form of bounding box loss with probabilities sigmoid for 1 and 0 values too.

Overall, the paper showed that it can be more effective to convert point clouds into voxels and apply Conv3D operations on a standardized spatial representation, trading some point-level geometric precision for efficient spatial feature extraction. The entire architecture is trained end to end and the final results obtain State of the art at the time on KITTI lidar dataset. The results are better partly because of obtaining features from a point net like layer from the voxel and thus it being extracted directly from data with thus reducing the reliance on hand crafted features.