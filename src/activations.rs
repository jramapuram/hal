use af;
use af::{Array};
use error::HALError;

/// Returns the tanh activated value
pub fn tanh(x: &Array) -> Array {
  af::tanh(x).unwrap()
}

/// Returns the sigmoid activated value
/// 1.0/(1.0 + exp(-1.0 * e))
pub fn sigmoid(x: &Array) -> Array {
  let denominator = af::add(&1.0f32, &af::exp(&af::mul(&-1.0f32, x, false).unwrap()).unwrap(), false).unwrap();
  af::div(&1.0f32, &denominator, false).unwrap()
}

/// Return the softmax activated value
/// exp(x_i) / sum(exp(x))
pub fn softmax(x: &Array) -> Array {
  let exponentiated = af::exp(&x).unwrap();
  af::div(&exponentiated, &af::sum_all(&exponentiated).unwrap().0, false).unwrap()
}

/// Return the lrelu activated value
/// max(0.01*x, x)
pub fn lrelu(x: &Array) -> Array {
  let scaled = af::mul(x, &0.0f32, false).unwrap();
  af::select(&scaled                                 // return 0.01x
             , &af::lt(x, &0.0f32, false).unwrap()   // if x > 0.0
             , x).unwrap().cast::<f32>().unwrap()    // else x
}

/// Return the derivative of relu [assumes that relu has already been applied]
/// 0.01 * x for x <= 0
/// 1        otherwise
pub fn lrelu_derivative(x: &Array) -> Array {
  let remove_negatives = af::selectl(0.01, &af::lt(x, &0.0f32, true).unwrap(), x).unwrap();
  af::selectl(1.0, &af::gt(x, &0.0f32, false).unwrap(), &remove_negatives).unwrap()
                                                                            .cast::<f32>()
                                                                            .unwrap()
}

/// Return the relu activated value
/// max(0, x)
pub fn relu(x: &Array) -> Array {
  af::selectl(0.0, &af::lt(x, &0.0f32, false).unwrap(), x).unwrap()
                                                         .cast::<f32>()
                                                         .unwrap()
}

/// Return the derivative of relu [assumes that relu has already been applied]
/// 0 for x <= 0
/// 1 otherwise
pub fn relu_derivative(x: &Array) -> Array {
  let zero_vec = af::constant(0.0f32, x.dims().unwrap()).unwrap();
  af::selectl(1.0f64, &af::gt(x, &0.0f32, false).unwrap(), &zero_vec).unwrap()
                                                                     .cast::<f32>()
                                                                     .unwrap()
}

/// Return the derivative of tanh [assumes that tanh has already been applied]
/// 1 - tanh(x)*tanh(x)
pub fn tanh_derivative(x: &Array) -> Array {
  af::sub(&1.0f32, &af::mul(x, x, false).unwrap(), false).unwrap()
}

/// Returns the derivative of sigmoid [assumes that sigmoid is already applied]
/// x * (1 - x)
pub fn sigmoid_derivative(x: &Array) -> Array {
  af::mul(x, &af::sub(&1.0f32, x, false).unwrap(), false).unwrap()
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
  af::constant(1.0f32, x.dims().unwrap()).unwrap()
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
