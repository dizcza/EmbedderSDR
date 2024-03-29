# Embedder SDR

EmbedderSDR is a family of neural network encoders that transform input images into binary Sparse Distributed Representation ([SDR](https://discourse.numenta.org/t/sparse-distributed-representations/2150)).

The goal of this project is gradient-free optimization by sparse vectors association in unsupervised fashion (refer to [Willshaw's model, 1969](https://redwood.berkeley.edu/wp-content/uploads/2018/08/willshaw1969.pdf)). While gradient-free optimization is not achieved yet, here I demonstrate how to construct meaningful (feature preserved, distributed) binary sparse vectors in deep learning with TripletLoss.

### Sparse Distributed Representation

Binary sparse distributed representations are formed with [k-winners-take-all](https://en.wikipedia.org/wiki/Winner-take-all_\(computing\)) activation function (kWTA) and subsequent binarization. It replaces the last softmax layer in neural networks. kWTA layer implementation is [here](models/kwta.py).


### Ideas

Below is the list of ideas, taken from neuroscience and applied to deep learning. Each item is marked with the symbol :heavy_check_mark: (true), :negative_squared_cross_mark: (false), or :black_square_button: (to be investigated).   

:heavy_check_mark: kWTA can be used in deep neural networks, showing competitive results with traditional softmax cross-entropy loss.

:negative_squared_cross_mark: Models with kWTA can be trained from a randomly initialized point in parameters space. This is true only for simple datasets such as MNIST, CIFAR10.

:heavy_check_mark: Models with kWTA can be trained from a model, pretrained on the same dataset with conventional cross-entropy loss (or any other loss you might like). You replace the last layer with a fully-connected, followed by kWTA.

:negative_squared_cross_mark: All ReLU activation functions can be replaced with kWTA. This is not true. kWTA is not meant to replace ReLU. It's not meant to pass the gradients either. Doing so loses model's capability for training. kWTA should be used only in the last layer.

:negative_squared_cross_mark: Trained on same-same class examples with Contrastive loss, embedding representations collapse to a constant vector ([Yann LeCun, 2006](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)). If we apply kWTA without changing the gradient flow and without telling what the class is, will it force the representation of each class to be unique?

:heavy_check_mark: [Synaptic scaling](https://en.wikipedia.org/wiki/Synaptic_scaling) helps. Synaptic scaling makes sure there are no dead neurons (which never participate in sensory coding) and weakens over-stimulated neurons (which are always active). Doing so increases the entropy of neural responses and thus the ability to preserve information about the stimuli.

:black_square_button: The latent space of binary SDR disentangles underlying factors of variation with simple operations from set theory: union, intersection, and difference. For example, an image of the dog `0010001000011` on the grass `0000101001010` would be a union of their representations - `0010101001011`. At the same time, a dog's SDR might consist of its underlying representations (fur, feet, tail, etc.). Bruno Olshausen gave a talk on this topic [here](https://youtu.be/QrvK3jPRc8k?t=3208).  


### Results

Below is the comparison between the traditional softmax and kWTA activation functions for `MLP_kWTA` shallow model, trained on MNIST dataset with the same optimizer and learning rate for both regimes. Cross-entropy loss is used for softmax, and triplet loss - for kWTA.

```
MLP_kWTA(
  (mlp): Sequential(
    (0): Linear(in_features=784, out_features=64, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=64, out_features=512, bias=True)
  )
  (kwta): KWinnersTakeAllSoft(sparsity=0.05)
)
```

1. Traditional softmax.

   In the traditional softmax model above the last activation function, kWTA, is replaced by ReLU.

   ![](images/softmax.png)

2. K-winners-take-all.

   The number of active neurons in the last layer with 512 neurons and kWTA activation function with the sparsity of 0.05 is `0.05 * 512 = 25`.

   ![](images/kwta.png)

   On the middle plot, the output of kWTA layer forms binary sparse distributed representation, averaged across samples for each class (label). Some neurons may respond to different patterns, but their ensemble activation uniquely encodes a stimulus (distributed coding).

   On the right plot, the Mutual Information Plane ([Tishby et al., 2017](https://arxiv.org/abs/1703.00810)) shows the convergence of the training process.

To reproduce the plots, call `train_kwta()` function in [`main.py`](main.py).

The complete infographics of the training progress is available on http://visdom.kyivaigroup.com:8097/. Give your browser a few minutes to parse the json data. Choose _"2020.07.28 MLP-kWTA: MNIST TrainerEmbeddingKWTA TripletLoss"_ environment.

### Papers, used in the code

This section is moved to https://github.com/dizcza/pytorch-mighty.
