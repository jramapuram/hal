# HAL : Hyper Adaptive Learning
Rust based Cross-GPU Machine Learning. 

# Why Rust? 
This project is for those that miss strongly typed compiled languages.                                                
Rust was chosen specifically because of [this](http://www.oreilly.com/programming/free/files/why-rust.pdf)

# Features
  
  - Multi GPU [model based] support
  - **OpenCL + CUDA + CPU build support**
  - **LSTM's with internal RTRL**
  - Perceptron's
  - Activations:     [Sigmoid, Tanh, Softmax]
  - Initializations: [Lecun Uniform, Glorot Normal, Glorot Uniform]
  - Loss Functions:  [MSE, Cross-Entropy]
  - OpenGL based plotting
  - AutoEncoder's **[Work in Progress]**
  - Multi GPU [horizontal] support **[TODO]**
  
# Installation
See [here](docs/installation.md)

# Credits
  - Thanks to the [arrayfire](http://arrayfire.com/) team for working with me to get the [rust bindings](https://github.com/arrayfire/arrayfire-rust) up.
  - [Keras](https://github.com/fchollet/keras) for inspiration as a lot of functions are similar to their implementation (minus the theano nonsense).
  - Dr. Felix Gers for his insight into internal RTRL
  - Dr. Sepp Hochreiter for advise on LSTM's
