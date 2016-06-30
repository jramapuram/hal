use af;
use af::Array;

use utils;
use error::HALError;

/// Return a vector form of the l2 error
/// (y - x) * (y - x)
pub fn l2_vec(pred: &Array, target: &Array) -> Array{
  let diff = af::sub(pred, target, false);
  af::mul(&diff, &diff, false)
}

/// Return a vector form of the mean squared error
/// 0.5 * (y - x) * (y - x)
pub fn mse_vec(pred: &Array, target: &Array) -> Array {
  af::mul(&l2_vec(pred, target), &0.5f32, false)
}

/// Return a vector form of cross entropy
/// -ylnx - [1-y]ln[1-x]
pub fn cross_entropy_vec(pred: &Array, target: &Array) -> Array {
  // -yln x
  let pos = af::mul(&af::mul(&-1.0f32, target, false)
                    , &af::log(&utils::clip_by_value(pred, 1e-10, 1.0)), false);
  //[1-y]ln[1-x]
  let neg = af::mul(&af::sub(&1.0f32, target, false)
                    , &af::log(&utils::clip_by_value(&af::sub(&1.0f32
                                                              , pred
                                                              , false)
                                                     , 1e-10, 1.0)), false);
  af::sub(&pos, &neg, false)
}

/// Provide a reduced form the L2 loss (single scalar)
pub fn l2(pred: &Array, target: &Array) -> f32 {
  af::sum_all(&l2_vec(pred, target)).0 as f32
}

/// Provide a reduced form the mean squared error loss (single scalar)
pub fn mse(pred: &Array, target: &Array) -> f32 {
  0.5f32 * af::mean_all(&l2_vec(pred, target)).0 as f32
}

/// Provide a reduced form the cross-entropy loss (single scalar)
pub fn cross_entropy(pred: &Array, target: &Array) -> f32 {
  // y: true target, x: prediction
  // E = sum(-ylnx - [1-y]ln[1-x])
  af::sum_all(&cross_entropy_vec(pred, target)).0 as f32
}

/// Provides the vector derivative of the mean squared error
pub fn mse_derivative(pred: &Array, target: &Array) -> Array {
  af::sub(pred, target, false)
}
/// Provides the vector derivative of the l2 error
pub fn l2_derivative(pred: &Array, target: &Array) -> Array {
  af::mul(&mse_derivative(pred, target), &2.0f32, false)
}

/// Provides the vector derivative of the cross-entropy error
pub fn cross_entropy_derivative(pred: &Array, target: &Array) -> Array {
  mse_derivative(pred, target)
}

/// Helper to provide a loss from a string
pub fn get_loss(name: &str, pred: &Array, target: &Array) -> Result<f32, HALError> {
  match name {
    "l2"            => Ok(l2(pred, target)),
    "mse"           => Ok(mse(pred, target)),
    "cross_entropy" => Ok(cross_entropy(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}

/// Helper to provide a loss vector from a string
pub fn get_loss_vec(name: &str, pred: &Array, target: &Array) -> Result<Array, HALError> {
  match name {
    "l2"            => Ok(l2_vec(pred, target)),
    "mse"           => Ok(mse_vec(pred, target)),
    "cross_entropy" => Ok(cross_entropy_vec(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}

/// Helper to provide a loss derivative from a string
pub fn get_loss_derivative(name: &str, pred: &Array, target: &Array) -> Result<Array, HALError> {
  match name {
    "l2"            => Ok(l2_derivative(pred, target)),
    "mse"           => Ok(mse_derivative(pred, target)),
    "cross_entropy" => Ok(cross_entropy_derivative(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}
