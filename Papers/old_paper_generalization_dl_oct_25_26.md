# Understanding Deep Learning Requires Rethinking Generalization

A claim-based testing paper.

**Full paper:** Understanding Deep Learning Requires Rethinking Generalization  
**Authors:** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals  
**Published:** ICLR 2017  
**Project page:** Not available  
**Link:** https://arxiv.org/abs/1611.03530

## Paper Summary: Claim by Claim

### Claim 1: Neural networks can memorize arbitrary training data

**Method:** Train standard neural network architectures on datasets with randomly assigned labels and on images replaced with random noise.

**Result:** The networks achieve nearly zero training error, demonstrating that they can memorize arbitrary training examples even when meaningful input-label relationships are absent.

### Claim 2: Traditional capacity measures struggle to explain generalization

**Method:** Analyze the randomized-label experiments in relation to VC dimension, Rademacher complexity, and uniform stability.

**Result:** Because the tested networks can fit random labels, conventional capacity measures can produce bounds too loose to explain why the same architectures generalize well when trained on real labels. The authors argue that these measures are insufficient as explanations in the settings studied.

### Claim 3: Explicit regularization is not the fundamental explanation for generalization

**Method:** Compare model performance with and without weight decay, dropout, and data augmentation. Also test whether regularization prevents memorization of random labels.

**Result:** Networks can generalize reasonably well without explicit regularization, although regularization often improves test performance. Some networks can also fit random labels despite regularization. The authors conclude that explicit regularization is neither necessary nor sufficient, by itself, to explain generalization.

### Claim 4: Small neural networks can memorize finite datasets in principle

**Method:** Construct a two-layer ReLU network with \(2n+d\) parameters, where \(n\) is the number of distinct training examples and \(d\) is the input dimension.

**Result:** The authors theoretically demonstrate that such a network can represent any assignment of labels to those \(n\) examples. This establishes finite-sample expressivity, not that SGD will necessarily find every possible assignment.

### Claim 5: The optimization algorithm may influence generalization

**Method:** Analyze SGD in linear models and examine which solutions it selects when multiple solutions fit the training data perfectly.

**Result:** SGD can implicitly favour minimum-norm solutions without explicit regularization. However, minimum norm alone does not adequately predict generalization, so this does not provide a complete explanation for deep neural networks.