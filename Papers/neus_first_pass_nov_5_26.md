# NeuS (First Pass)

**Full paper:** NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

**Authors:** Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, Wenping Wang

**Published:** NeurIPS 2021

**Project page:** https://lingjie0206.github.io/papers/NeuS/

**Link:** https://arxiv.org/abs/2106.10689

The main goal for the paper: Using multiple calibrated cameras (calibrate with COLMAP before) create reconstruction in 3D. This is a bit similar to Nerfs, though rather than focusing on new views here the focus is on creating proper surface. To account for this, they are focusing on using SDF as the output representation rather than density. With color being another value being used for view to image matching. The main contribution apart from new representation experiment is new rendering system for this representation.

For the rendering part, they derive a way to perform volume rendering using the SDF instead of directly predicting density, while trying to avoid the geometric bias that comes with conventional volume rendering. They also introduce Eikonal regularization to encourage the network to learn a proper SDF.

One question that I have though is Eikonal regularization seems to have a much larger impact on SDF accuracy than on the actual reconstructed surface. Since the end goal here is reconstruction, and in abelation they show that even without eikonal regularization chamfer value holds up, I am not really sure why we even need it? We could always extract the mesh and calculate distances from it afterwards for better SDF later on. Mesh could be generated using this representation anyways using marching cubes.

NOTE: Need to spend some more time understanding the maths by manually deriving everything by hand, but the main idea and main concepts I have already understood. Would need to give another pass to this paper later on.