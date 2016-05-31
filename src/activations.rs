use af;
use af::{Array, HasAfEnum, Aftype};
use num::Complex;

use utils;
use error::HALError;

/// Returns the tanh activated value
pub fn tanh(x: &Array) -> Array {
  af::tanh(x).unwrap()
}

/// Returns the sigmoid activated value
/// 1.0/(1.0 + exp(-1.0 * e))
pub fn sigmoid(x: &Array) -> Array {
  let neg_one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), -1.0f32);
  let one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 1.0f32);
  let exp_m_e = af::exp(&af::mul(&neg_one, x, false).unwrap()).unwrap();
  let denominator = af::add(&one, &exp_m_e, false).unwrap();
  let a = af::div(&one, &denominator, false).unwrap();
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the softmax activated value
/// exp(x_i) / sum(exp(x))
pub fn softmax(x: &Array) -> Array {
  let exponentiated = af::exp(&x).unwrap();
  let sum_epx_x = af::sum_all(&exponentiated).unwrap().0 as f32;
  let sum_exp_x_vec = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), sum_epx_x);
  let a = af::div(&exponentiated, &sum_exp_x_vec, false).unwrap();
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the lrelu activated value
/// max(0.01*x, x)
pub fn lrelu(x: &Array) -> Array {
  let zero_one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 0.01f32);
  let scaled = af::mul(x, &zero_one, false).unwrap();
  let a = af::select(&scaled                               // return 0.01x
                     , &af::lt(x, &0.0f32, false).unwrap() // if x > 0.0
                     , x).unwrap();                        // else x
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the derivative of relu [assumes that relu has already been applied]
/// 0.01     for x <= 0
/// 1        otherwise
pub fn lrelu_derivative(x: &Array) -> Array {
  let x_lt_zero = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 0.01f32);
  let one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 1.0f32);
  let grad = af::select(&one, &af::gt(x, &0.0f32, false).unwrap(), &x_lt_zero).unwrap();
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Return the relu activated value
/// max(0, x)
pub fn relu(x: &Array) -> Array {
  let zero = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 0.0f32);
  let a = af::select(&zero, &af::lt(x, &0.0, false).unwrap(), x).unwrap();
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the derivative of relu [assumes that relu has already been applied]
/// 0 for x <= 0
/// 1 otherwise
pub fn relu_derivative(x: &Array) -> Array {
  let zero = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 0.0f32);
  let one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 1.0f32);
  let grad = af::select(&one, &af::gt(x, &0.0f32, false).unwrap(), &zero).unwrap();
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Return the derivative of tanh [assumes that tanh has already been applied]
/// 1 - tanh(x)*tanh(x)
pub fn tanh_derivative(x: &Array) -> Array {
  let one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 1.0f32);
  let grad = af::sub(&one, &af::mul(x, x, false).unwrap(), false).unwrap();
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Returns the derivative of sigmoid [assumes that sigmoid is already applied]
/// x * (1 - x)
pub fn sigmoid_derivative(x: &Array) -> Array {
  let one = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 1.0f32);
  let grad = af::mul(x, &af::sub(&one, x, false).unwrap(), false).unwrap();
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Returns the derivative of softmax [assumes that it was already applied]
/// x * (1 - x)
pub fn softmax_derivative(x: &Array) -> Array {
  sigmoid_derivative(x)
}

/// Returns a linear activation [no non-linearity]
pub fn ones(x: &Array) -> Array {
  x.clone()
}

/// Returns a derivative of a linear activation (1's)
pub fn ones_derivative(x: &Array) -> Array {
  let grad = utils::constant(x.dims().unwrap(), x.get_type().unwrap(), 1.0f32);
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Helper to determine whether function is smooth or non-smooth
pub fn is_smooth(name: &str) -> bool {
  match name {
    "softmax" => true,
    "sigmoid" => true,
    "relu"    => false,
    "lrelu"   => false,
    "tanh"    => true,
    "ones"    => true,
    "linear"  => true,
    _         => panic!("unknown function name provided"),
  }
}

/// Helper to get the correct activation using a string
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

/// Helper to get the correct activation derivative using a string
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
