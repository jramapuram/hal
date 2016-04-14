# HAL : Hyper Adaptive Learning
Rust based Cross-GPU Machine Learning. [![Build Status](https://travis-ci.org/jramapuram/hal.svg?branch=feature%2Flstm)](https://travis-ci.org/jramapuram/hal)

# Why Rust?
This project is for those that miss strongly typed compiled languages.
Rust was chosen specifically because of [this](http://www.oreilly.com/programming/free/files/why-rust.pdf)
Furthermore, we can offer fine grained control of your operations.
This means being able to grab dimensions of tensors at any stage, none of that unknown shape nonsense.
We can also micro-control steps. An example of this working with each individual forward timesteps on say an LSTM. Usually these are controlled by inner loops [Theano/Tensorflow, etc].

# Features
  - Multi GPU [model based] support
  - **OpenCL + CUDA + Parallel CPU support**
  - **LSTM's with internal RTRL [Work in Progress]**
  - **RNN's [Work in Progress]**
  - Perceptrons, AutoEncoders, ConvNets [TODO]
  - Optimizers:      [SGD, Adam[TODO], AdaGrad[TODO]]
  - Activations:     [Linear, Sigmoid, Tanh, ReLU, LReLU, Softmax]
  - Initializations: [Lecun Uniform, Glorot Normal, Glorot Uniform, Normal, Uniform]
  - Data Gatherers:  [SinSource, MNIST[TODO], CIFAR10[TODO]]
  - Loss Functions:  [MSE, L2, Cross-Entropy]
  - OpenGL based plotting and image loading, see [here](https://www.accelereyes.com/arrayfire/c/page_gfx.htm) for more info
  - Multi GPU [horizontal] support **[TODO]**

# Installation
See [here](docs/installation.md)

# Testing
HAL utilizes RUST's test framework to test all of our modules.
We employ gradient checking on individual functions as well as layers.
Testing needs to run on one thread due to our device probing methods.
```bash
export RUST_TEST_THREADS=1
cargo test
```

# Credits
  - Thanks to the [arrayfire](http://arrayfire.com/) team for working with me to get the [rust bindings](https://github.com/arrayfire/arrayfire-rust) up.
  - [Keras](https://github.com/fchollet/keras) for inspiration as a lot of functions are similar to their implementation (minus the theano nonsense).
  - Dr. Felix Gers for his insight into internal RTRL
  - Dr. Sepp Hochreiter for advise on LSTM's
