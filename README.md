# HAL : Hyper Adaptive Learning
Rust based Cross-GPU Machine Learning. 

# Why Rust? 
This project is for those that miss strongly typed compiled languages (where for-loops are okay!)
**Note:** We still vectorize all the required BLAS tasks on the GPU.

# Features
  
  - Multi GPU support
  - OpenCL + CUDA + CPU build support
  - Perceptron's
  - Activations:     [Sigmoid, Tanh, Softmax]
  - Initializations: [Lecun Uniform, Glorot Normal, Glorot Uniform]
  - Loss Functions:  [MSE, Cross-Entropy]
  - OpenGL based plotting
  - AutoEncoder's **[Work in Progress]**
  - LSTM's        **[Work in Progress]**
  - RTRL          **[Work in Progress]**
  
# Installation
See [here](docs/installation.md)

# Credits
  - Thanks to the [arrayfire](http://arrayfire.com/) team for working with me to get the [rust bindings](https://github.com/arrayfire/arrayfire-rust) up.
  - [Keras](https://github.com/fchollet/keras) for inspiration as a lot of functions are similar to their implementation (minus the theano nonsense).
