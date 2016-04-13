extern crate hal;
extern crate arrayfire as af;

use af::{Array, Dim4};

use hal::{utils, activations, initializations, loss};


fn verify_derivative<F>(ufunc: F, name: &str, kinks: bool)
  where F : Fn(&Array) -> Array
{
  println!("\nGradient testing {}...", name);
  let dims = Dim4::new(&[32, 1, 1, 1]);
  let x = initializations::normal(dims, 0.0f32);
  let grad = activations::get_derivative(name, &ufunc(&x)).unwrap();
  match kinks {
    true  => utils::verify_gradient_kinks(ufunc, &x, 1e-5, &grad).unwrap(),
    false => utils::verify_gradient_smooth(ufunc, &x, 1e-5, &grad).unwrap(),
  };
}

fn verify_func<F>(ufunc: F, name: &str, truth: &[f32])
  where F : Fn(&Array) -> Array
{
  println!("\nTesting unitary function {}...", name);
  let dims = Dim4::new(&[4, 1, 1, 1]);
  let x = Array::new(&[-1.0, 1.0, 2.0, 3.0], dims).unwrap();

  // verify with L2 Loss
  let x_t = Array::new(truth, dims).unwrap();
  let l2 = loss::get_loss("l2", &ufunc(&x), &x_t).unwrap();
  assert!(l2 <= 1e-4, "L2 loss of {} is higher than expected: {}", name, l2);
}


#[test]
fn lrelu_gradient() {
  verify_derivative(activations::lrelu
                    , "lrelu"
                    , true);
}

#[test]
fn tanh_gradient() {
  verify_derivative(activations::tanh
                    , "tanh"
                    , false);
}

#[test]
fn sigmoid_gradient() {
  verify_derivative(activations::sigmoid
                    , "sigmoid"
                    , false);
}

#[test]
fn softmax_gradient() {
  verify_derivative(activations::softmax
                    , "softmax"
                    , false);
}

#[test]
fn relu_gradient() {
  verify_derivative(activations::relu
                    , "relu"
                    , true);
}

#[test]
fn ones_gradient() {
  verify_derivative(activations::ones
                    , "ones"
                    , false);
}


/// Test unitary functions

#[test]
fn tanh(){
  verify_func(activations::tanh
              , "tanh"
              , &[-0.7616, 0.7616, 0.9640, 0.9951]);
}

#[test]
fn sigmoid(){
  verify_func(activations::sigmoid
              , "sigmoid"
              , &[0.2689, 0.7311, 0.8808, 0.9526]);
}

#[test]
fn relu(){
  verify_func(activations::relu
              , "relu"
              , &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn lrelu(){
  verify_func(activations::lrelu
              , "lrelu"
              , &[-0.01, 1.0, 2.0, 3.0]);
}

#[test]
fn softmax(){
  verify_func(activations::softmax
              , "softmax"
              , &[0.0120, 0.0889, 0.2418, 0.6572]);
}

#[test]
fn ones(){
  verify_func(activations::ones
              , "ones"
              , &[-1.0, 1.0, 2.0, 3.0]);
}
