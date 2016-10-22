use af;
use af::{Array};
use num::Complex;

use utils;
use error::HALError;

/// Returns the tanh activated value
pub fn tanh(x: &Array) -> Array {
  af::tanh(x)
}

/// Returns the sigmoid activated value
/// 1.0/(1.0 + exp(-1.0 * e))
pub fn sigmoid(x: &Array) -> Array {
  let neg_one = utils::constant(x.dims(), x.get_type(), -1.0f32);
  let one = utils::constant(x.dims(), x.get_type(), 1.0f32);
  let exp_m_e = af::exp(&af::mul(&neg_one, x, false));
  let denominator = af::add(&one, &exp_m_e, false);
  let a = af::div(&one, &denominator, false);
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the softmax activated value [numerical stable]
/// exp(x_i) / sum(exp(x))
pub fn softmax(x: &Array) -> Array {
  // http://www.deeplearningbook.org/contents/mlp.html page 185
  let z = match x.numdims() {
    1 => x.clone(),
    _ => af::sub(x, &af::max(x, 1), true),
  };

  let exponentiated = af::exp(&z);
  let sum_exp_x_vec = af::sum(&exponentiated, 1);
  let a = af::div(&exponentiated, &sum_exp_x_vec, true);
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the lrelu activated value
/// max(0.01*x, x)
pub fn lrelu(x: &Array) -> Array {
  let zero_one = utils::constant(x.dims(), x.get_type(), 0.01f32);
  let scaled = af::mul(x, &zero_one, false);
  let a = af::select(&scaled                               // return 0.01x
                     , &af::lt(x, &0.0f32, false) // if x > 0.0
                     , x);                        // else x
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the derivative of relu [assumes that relu has already been applied]
/// 0.01     for x <= 0
/// 1        otherwise
pub fn lrelu_derivative(x: &Array) -> Array {
  let x_lt_zero = utils::constant(x.dims(), x.get_type(), 0.01f32);
  let one = utils::constant(x.dims(), x.get_type(), 1.0f32);
  let grad = af::select(&one, &af::gt(x, &0.0f32, false), &x_lt_zero);
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Return the relu activated value
/// max(0, x)
pub fn relu(x: &Array) -> Array {
  let zero = utils::constant(x.dims(), x.get_type(), 0.0f32);
  let a = af::select(&zero, &af::lt(x, &0.0, false), x);
  utils::assert_types(vec![x, &a]);
  a
}

/// Return the derivative of relu [assumes that relu has already been applied]
/// 0 for x <= 0
/// 1 otherwise
pub fn relu_derivative(x: &Array) -> Array {
  let zero = utils::constant(x.dims(), x.get_type(), 0.0f32);
  let one = utils::constant(x.dims(), x.get_type(), 1.0f32);
  let grad = af::select(&one, &af::gt(x, &0.0f32, false), &zero);
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Return the derivative of tanh [assumes that tanh has already been applied]
/// 1 - tanh(x)*tanh(x)
pub fn tanh_derivative(x: &Array) -> Array {
  let one = utils::constant(x.dims(), x.get_type(), 1.0f32);
  let grad = af::sub(&one, &af::mul(x, x, false), false);
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Returns the derivative of sigmoid [assumes that sigmoid is already applied]
/// x * (1 - x)
pub fn sigmoid_derivative(x: &Array) -> Array {
  let one = utils::constant(x.dims(), x.get_type(), 1.0f32);
  let grad = af::mul(x, &af::sub(&one, x, false), false);
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Returns the derivative of softmax [assumes that it was already applied]
/// x * (1 - x)
pub fn softmax_derivative(x: &Array) -> Array {
  // let orig_dims = x.dims();
  // let slide_dims = orig_dims;
  // slide_dims[2] = slide_dims[1];
  // slide_dims[1] = 1;
  // let j = af::mul(x, &af::moddims(x, slide_dims));
  // let diag = sigmoid_derivative(x);
  // af::print(&af::diag_extract(&j, 1))
  sigmoid_derivative(x)
}

/// Returns a linear activation [no non-linearity]
pub fn ones(x: &Array) -> Array {
  x.clone()
}

/// Returns a derivative of a linear activation (1's)
pub fn ones_derivative(x: &Array) -> Array {
  let grad = utils::constant(x.dims(), x.get_type(), 1.0f32);
  utils::assert_types(vec![x, &grad]);
  grad
}

/// Apply relu on the module of the complex array
pub fn mod_relu(z: Array, b: Array) -> Array {
  let module_z = af::root(&2f32, &af::real(&af::mul(&af::conjg(&z), &z, false)), true);
  let new_module_z = self::get_activation("relu", &af::add(&module_z, &b, true)).unwrap();
  af::div(&af::mul(&z, &new_module_z, false), &module_z, false)
}

/// Compute mod_relu derivative with respect to z
pub fn mod_relu_derivative_z(z: Array, b: Array, d_h: Array) -> Array {
  let module_carre_z = af::real(&af::mul(&af::conjg(&z), &z, false));
  let module_z = af::root(&2f32, &module_carre_z, true);
  let new_module_z = self::get_activation("relu", &af::add(&module_z, &b, true)).unwrap();

  let d_activ = self::get_derivative("relu", &new_module_z).unwrap();
  let mut d_z1 = af::div(&af::mul(&d_h, &af::conjg(&z), false), &module_z, false);
  d_z1 = af::mul(&d_z1, &d_activ, false);
  d_z1 = af::div(&d_z1, &module_z, false);
  d_z1 = af::div(&d_z1, &2f32, false);
  d_z1 = af::add(&af::mul(&z, &d_z1, false), &af::conjg(&af::mul(&af::conjg(&z), &d_z1, false)), false);

  let mut d_z2 = af::mul(&d_h, &new_module_z, false);
  d_z2 = af::mul(&d_z2, &af::conjg(&z), false);
  d_z2 = af::div(&d_z2, &module_carre_z, false);
  d_z2 = af::div(&d_z2, &module_z, false);
  d_z2 = af::div(&d_z2, &2f32, false);
  d_z2 = af::add(&af::mul(&z, &d_z2, false), &af::conjg(&af::mul(&af::conjg(&z), &d_z2, false)), false);

  let mut d_z3 = af::mul(&d_h, &new_module_z, false);
  d_z3 = af::div(&d_z3, &module_z, false);

  af::add(&af::sub(&d_z1, &d_z2, false), &d_z3, false)
}

/// Compute mod_relu derivative with respect to b
pub fn mod_relu_derivative_b(z: Array, b: Array, d_h: Array) -> Array {
  let module_z = af::root(&2f32, &af::real(&af::mul(&af::conjg(&z), &z, false)), true);
  let new_module_z = self::get_activation("relu", &af::add(&module_z, &b, true)).unwrap();
  let d_activ = self::get_derivative("relu", &new_module_z).unwrap();

  let d_b = af::div(&af::mul(&d_h, &af::conjg(&z), false), &module_z, false);
  af::mul(&d_b, &d_activ, false)
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
