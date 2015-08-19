use af;
use af::{Dim4, Array};
use error::HALError;

pub fn tanh(x: &Array) -> Array {
  af::tanh(x).unwrap()
}

pub fn sigmoid(x: &Array) -> Array {
  // let exponentiated = x.map(|e| 1.0/(1.0 + af::exp(-1.0 * e)));
  // exponentiated.unwrap()
  af::div(1.0, af::add(1.0, af::exp(af::mul(-1.0, x)))).unwrap()
}

pub fn softmax(x: &Array) -> Array {
  let exponentiated = af::exp(x).unwrap();
  // let exponentialted_sum = af::sum(exponentiated).unwrap();
  // let smax = exponentiated.map(|elem| af::div(elem, exponentialted_sum)).unwrap();
  // smax
  af::div(exponentiated, af::sum(exponentiated)).unwrap()
}

pub fn tanh_derivative(x: &Array) -> Array {
  af::sub(1.0, af::mul(x, x)).unwrap()
}

pub fn sigmoid_derivative(x: &Array) -> Array {
  af::mul(x, af::sub(1.0, x)).unwrap()
}

pub fn softmax_derivative(x: &Array) -> Array {
  //TODO: Verify
  sigmoid_derivative(x)
}

pub fn get_activation(name: &str, x: &Array) -> Array {
  match(name){
    "softmax" => softmax(x),
    "sigmoid" => sigmoid(x),
    "tanh"    => tanh(x),
    _         => HALError::UNKNOWN,
  }
}

pub fn get_activation_derivative(name: &str, x: &Array) -> Array {
  match(name){
    "softmax" => softmax_derivative(x),
    "sigmoid" => sigmoid_derivative(x),
    "tanh"    => tanh_derivative(x),
    _         => HALError::UNKNOWN,
  }  
}
