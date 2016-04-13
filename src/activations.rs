use af;
use af::{Array, Aftype};
use error::HALError;

pub fn tanh(x: &Array) -> Array {
  af::tanh(x).unwrap()
}

pub fn sigmoid(x: &Array) -> Array {
  // let exponentiated = x.map(|e| 1.0/(1.0 + af::exp(-1.0 * e)));
  // exponentiated.unwrap()
  let denominator = af::add(&1.0f32, &af::exp(&af::mul(&-1.0f32, x, false).unwrap()).unwrap(), false).unwrap();
  af::div(&1.0f32, &denominator, false).unwrap()
}

pub fn softmax(x: &Array) -> Array {
  let exponentiated = af::exp(&x).unwrap();
  // let exponentialted_sum = af::sum(exponentiated).unwrap();
  // let smax = exponentiated.map(|elem| af::div(elem, exponentialted_sum)).unwrap();
  // smax
  af::div(&exponentiated, &af::sum_all(&exponentiated).unwrap().0, false).unwrap()
}

pub fn lrelu(x: &Array) -> Array {
  let scaled = af::mul(x, &0.0f32, false).unwrap();
  af::select(&scaled                                      // return 0.01x
             , &af::lt(x, &0.0f32, true).unwrap()         // if x > 0.0
             , x).unwrap().cast::<f32>().unwrap() // else x
}

/// Provides the derivative of lrelu
pub fn lrelu_derivative(x: &Array) -> Array {
  let remove_negatives = af::selectl(0.01, &af::lt(x, &0.0f32, true).unwrap(), x).unwrap();
  af::selectl(1.0, &af::gt(x, &0.0f32, true).unwrap(), &remove_negatives).unwrap()
                                                                            .cast::<f32>()
                                                                            .unwrap()
}

pub fn relu(x: &Array) -> Array {
  af::selectl(0.0, &af::lt(x, &0.0f32, true).unwrap(), x).unwrap()
                                                         .cast::<f32>()
                                                         .unwrap()
}

pub fn relu_derivative(x: &Array) -> Array {
  let remove_negatives = af::selectl(0.0, &af::lt(x, &0.0f32, true).unwrap(), x).unwrap();
  af::selectl(1.0, &af::gt(x, &0.0f32, true).unwrap(), &remove_negatives).unwrap()
                                                                            .cast::<f32>()
                                                                            .unwrap()
}

/// Return the derivative of tanh [assumes that tanh has already been applied]
/// 1 - tanh(x)*tanh(x)
///
pub fn tanh_derivative(x: &Array) -> Array {

  // let t = tanh(x);
  // af::sub(&1.0f32, &af::mul(&t, &t, false).unwrap(), false).unwrap()
  af::sub(&1.0f32, &af::mul(x, x, false).unwrap(), false).unwrap()
}

pub fn sigmoid_derivative(x: &Array) -> Array {
  // x * (1 - x)
  //let s = sigmoid(x);
  //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap()
  af::mul(x, &af::sub(&1.0f32, x, false).unwrap(), false).unwrap()
}

pub fn softmax_derivative(x: &Array) -> Array {
  // x * (1 - x)
  //let s = softmax(x);
  //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap()
  sigmoid_derivative(x)
}

pub fn ones(x: &Array) -> Array {
  x.clone()
}

pub fn ones_derivative(x: &Array) -> Array {
  af::constant(1.0f32, x.dims().unwrap()).unwrap()
}

pub fn get_activation(name: &str, x: &Array) -> Result<Array, HALError> {
  match name {
    "softmax" => Ok(softmax(x)),
    "sigmoid" => Ok(sigmoid(x)),
    "relu"    => Ok(relu(x)),
    "lrelu"   => Ok(lrelu(x)),
    "tanh"    => Ok(tanh(x)),
    "ones"    => Ok(ones(x)),
    "linear"  => Ok(ones(x)),
    _         => Err(HALError::UNKNOWN),
  }
}

pub fn get_derivative(name: &str, x: &Array) -> Result<Array, HALError> {
  match name {
    "softmax" => Ok(softmax_derivative(x)),
    "sigmoid" => Ok(sigmoid_derivative(x)),
    "relu"    => Ok(relu_derivative(x)),
    "lrelu"   => Ok(lrelu_derivative(x)),
    "tanh"    => Ok(tanh_derivative(x)),
    "ones"    => Ok(ones_derivative(x)),
    "linear"  => Ok(ones_derivative(x)),
    _         => Err(HALError::UNKNOWN),
  }
}
