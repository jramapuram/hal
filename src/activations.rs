use af::Dim4;
use af::Array;
use error::HALError;

pub fn tanh(x: &Array) -> Array {
  af::tanh(x).unwrap()
}

pub fn sigmoid(x: &Array) -> Array {
  let exponentiated = x.map(|e| 1.0/(1.0 + af::exp(-1.0 * e)));
  exponentiated.unwrap()
}

pub fn softmax(x: &Array) -> Array {
  let exponentiated = af::exp(x).unwrap();
  let exponentialted_sum = af::sum(exponentiated).unwrap();
  exponentiated.map(|elem| elem/exponentialted_sum).unwrap()
}

pub fn tanh_derivative(x: &Array) -> Array {
  let t = af::mul(-1.0, tanh(x))
  af::add(1.0, t).unwrap()
}

pub fn sigmoid_derivative(x: &Array) -> Array {
  let sigm = sigmoid(x);
  af::dot(sigm, af::add(1.0, af::mul(-1.0, sigm))).unwrap()
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
