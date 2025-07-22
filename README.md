# pytorch-GANs
PyTorch implementations of various GAN papers/architectures like GAN (Goodfellow at at.), DCGAN (Radford et al.), etc. 

## Implemented Papers

* [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)

* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al., 2016)](https://arxiv.org/abs/1511.06434)

## Generative Adversarial Networks (Goodfellow et al., 2014)
This code tries to reproduce the 2014 Goodfellow et al. paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

Original code and hyperparameters from the paper can be found [here](https://github.com/goodfeli/adversarial).

### MNIST (fully connected model)

#### Model Architecture:

Generator:
* Input layer: 100 units (Uniform[-1, 1] noise)
* Dense layer: 1200 units, ReLU
* Dense layer: 1200 units, ReLU
* Output layer: 784 units, Sigmoid

Discriminator:
* Input layer: 784 units (flattend 28x28 gray-scale image)
* Maxout layer: 240 units, 5 pieces
* Maxout layer: 240 units, 5 pieces
* Output layer: 1 unit, Sigmoid 

<div style="text-align: center;":>
      <img src="./res/mnist.gif" width="400">
</div>


| Epoch 001 | Epoch 100 |Loss |
| --------- | ---------- |-----------|
| ![mnist](./res/vanilla_gan_mnist_000.png) | ![mnist](./res/vannila_gan_mnist_100.png)| ![loss](./res/vannila_gan_loss_mnist.png)|

### CelebFaces (fully connected model)

#### Model Architecture:

Generator:
* Input layer:    100 units (Uniform[-1, 1] noise)
* Dense layer:    8000 units, ReLU
* Output layer:   2304 units, Sigmoid

Discriminator:
* Input layer: 2304 units (flattend 48x48 grayscale image)
* Maxout layer: 1200 units, 5 pieces
* Maxout layer: 1200 units, 5 pieces
* Output layer: 1 unit, Sigmoid 

<div style="text-align: center;":>
      <img src="./res/celebfaces_fc_gray.gif" width="400">
</div>

| Epoch 001 | Epoch 250 |Loss |
| --------- | ---------- |-----------|
| ![celebfaces_fc_vanilla](./res/vannila_gan_celebfaces_000.png) | ![celebfaces_fc_vanilla](./res/vannila_gan_celebfaces_100.png)| ![celebfaces_fc_loss_vanilla](./res/vannila_gan_loss_celebfaces.png)|


## Deep Convolutional Generative Adversarial Networks (Radford et al. 2016)
This code tries to reproduce the 2016 Radford et al. paper [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

#### CelebFaces (deep convolutional version)

<div style="text-align: center;":>
      <img src="./res/dcgan_celebfaces.gif" width="400">
</div>



| Epoch 01 | Epoch 50 | Loss |
| --------- | ---------- | ------- | 
| ![celebfaces_dcgan](./res/dcgan_celebfaces_00.png) | ![celebfaces_dcgan](./res/dcgan_celebfaces_50.png) | ![dcgan_loss_celeb](./res/dcgan_loss_cifar.png)|

## Citations

```bibtex
@misc{goodfellow2014generativeadversarialnetworks,
      title={Generative Adversarial Networks}, 
      author={Ian J. Goodfellow and Jean Pouget-Abadie and Mehdi Mirza and Bing Xu and David Warde-Farley and Sherjil Ozair and Aaron Courville and Yoshua Bengio},
      year={2014},
      eprint={1406.2661},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1406.2661}, 
}
```

```bibtex
@misc{radford2016unsupervisedrepresentationlearningdeep,
      title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks}, 
      author={Alec Radford and Luke Metz and Soumith Chintala},
      year={2016},
      eprint={1511.06434},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1511.06434}, 
}
```