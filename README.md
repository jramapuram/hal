# HAL : Hyper Adaptive Learning
Rust based Cross-GPU Machine Learning. [![Build Status](https://travis-ci.org/jramapuram/hal.svg?branch=feature%2Flstm)](https://travis-ci.org/jramapuram/hal) [![](http://meritbadge.herokuapp.com/hal-ml)](https://crates.io/crates/hal-ml)

## Why Rust?
This project is for those that miss strongly typed compiled languages.
Rust was chosen specifically because of [this](http://www.oreilly.com/programming/free/files/why-rust.pdf)
Furthermore, we can offer fine grained control of your operations.
This means being able to grab dimensions of tensors at any stage, none of that unknown shape nonsense.
We can also micro-control steps. An example of this working with each individual forward timesteps on say an LSTM. Usually these are controlled by inner loops [Theano/Tensorflow, etc].

## Features
  - Multi GPU [model based] support
  - **OpenCL + CUDA + Parallel CPU support**
  - **LSTM's with internal RTRL [Work in Progress]**
  - **RNN's [Work in Progress]**
  - Perceptrons, AutoEncoders, ConvNets**[TODO]**
  - Optimizers:      [SGD, Adam, AdaGrad**[TODO]**]
  - Activations:     [Linear, Sigmoid, Tanh, ReLU, LReLU, Softmax]
  - Initializations: [Lecun Uniform, Glorot Normal, Glorot Uniform, Normal, Uniform]
  - Data Gatherers:  [SinSource, MNIST**[In Progress]**, CIFAR10**[TODO]**]
  - Loss Functions:  [MSE, L2, Cross-Entropy]
  - OpenGL based plotting and image loading, see [here](https://www.accelereyes.com/arrayfire/c/page_gfx.htm) for more info
  - Multi GPU [horizontal] support **[TODO]**

## Requirements
  - Rust 1.8 +
  - [Download and install ArrayFire binaries](https://arrayfire.com/download) based on your operating
   system.

## Use from Crates.io [![](http://meritbadge.herokuapp.com/hal-ml)](https://crates.io/crates/hal-ml)

To use the rust bindings for ArrayFire from crates.io, the following requirements are to be met
first.

1. [Download and install ArrayFire binaries](https://arrayfire.com/download) based on your operating
   system.
2. Set the evironment variable `AF_PATH` to point to ArrayFire installation root folder.
3. Make sure you add the path to library files to your path environment variables.
    - On Linux & OSX: do `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib`
    - On Windows: Add `%AF_PATH%\lib` to your PATH environment variable.
4. Add hal-ml as a dependency in your Cargo.toml

## Build from Source

Edit [build.conf](arrayfire-rust/build.conf) to modify the build flags. The structure is a simple JSON blob. Currently Rust does not allow key:value pairs to be passed from the CLI. To use an existing ArrayFire installation modify the first three JSON values. You can install ArrayFire using one of the following two ways.

- [Download and install binaries](https://arrayfire.com/download)
- [Build and install from source](https://github.com/arrayfire/arrayfire)

To build arrayfire submodule available in the rust wrapper, you have to do the following.

```bash
git submodule update --init --recursive
cargo build
```
 This is recommended way to build Rust wrapper since the submodule points to the most compatible version of ArrayFire the Rust wrapper has been tested with. You can find the ArrayFire dependencies below.

- [Linux dependencies](http://www.arrayfire.com/docs/using_on_linux.htm)
- [OSX dependencies](http://www.arrayfire.com/docs/using_on_osx.htm)


## Examples
```bash
cargo run --example autoencoder
cargo run --example xor_rnn
```

## Testing
HAL utilizes RUST's test framework to extensively test all of our modules.  
We employ gradient checking on individual functions as well as layers.  
Testing needs to run on one thread due to our device probing methods.  
Furthermore, graphics **needs** to be disabled for testing [glfw issue].  
```bash
AF_DISABLE_GRAPHICS=1 RUST_TEST_THREADS=1 cargo test
```

If you would like to see the results of the test (as well as benchmarks) run:
```bash
AF_DISABLE_GRAPHICS=1 RUST_TEST_THREADS=1 cargo test -- --nocapture
```

## Credits
  - Thanks to the [arrayfire](http://arrayfire.com/) team for working with me to get the [rust bindings](https://github.com/arrayfire/arrayfire-rust) up.
  - [Keras](https://github.com/fchollet/keras) for inspiration as a lot of functions are similar to their implementation (minus the theano nonsense).
  - Dr. Felix Gers for his insight into internal RTRL
  - Dr. Sepp Hochreiter for advise on LSTM's
