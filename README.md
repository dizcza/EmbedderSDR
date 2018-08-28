# Embedder SDR

The idea was to take the advantage of binary sparse distributed representation (SDR) of embeddings in unsupervised learning in some domain such as computer vision. 

Trained on same-same class examples with Contrastive loss, embedding representations collapse to a constant vector ([Yann LeCun, 2006](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)). If we apply k-winners-take-all followed by binarization without changing the gradient flow, will it force the representation to be different for each class without telling what the class is (or, in any words, that samples are assigned to different classes)?

The short answer is _no_.

See the [results](http://ec2-34-227-113-244.compute-1.amazonaws.com:8099) for MNIST56 dataset (a subset of 5 and 6 digits of MNIST so that any can play with). Choose the environment for `EmbedderSDR` project. Watch for mutual information of `kwta` (k-winners-take-all) layer, which is the last layer in this NN.