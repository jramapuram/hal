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

#[test]
fn test_lrelu_gradient() {
  verify_derivative(activations::lrelu
                    , "lrelu"
                    , true);
}

#[test]
fn test_tanh_gradient() {
  verify_derivative(activations::tanh
                    , "tanh"
                    , false);
}

#[test]
fn test_sigmoid_gradient() {
  verify_derivative(activations::sigmoid
                    , "sigmoid"
                    , false);
}

#[test]
fn test_softmax_gradient() {
  verify_derivative(activations::softmax
                    , "softmax"
                    , false);
}

#[test]
fn test_relu_gradient() {
  verify_derivative(activations::relu
                    , "relu"
                    , true);
}

#[test]
fn test_ones_gradient() {
  verify_derivative(activations::ones
                    , "ones"
                    , false);
}
