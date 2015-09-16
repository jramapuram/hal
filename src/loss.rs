use af;
use af::Array;

use activations;
use error::HALError;

pub fn mse(pred: &Array, target: &Array) -> f32 {
  let diff = af::sub(pred, target, false).unwrap();
  (af::mean_all(&af::mul(&diff, &diff, false).unwrap()).unwrap()).0 as f32
}

pub fn cross_entropy(pred: &Array, target: &Array) -> f32 {
  // y: true target, x: prediction
  // E = sum(-ylnx - [1-y]ln[1-x])
  let pos = af::mul(&af::mul(&-1.0, target, false).unwrap(), &af::log(&pred).unwrap(), false).unwrap(); // -ylnx
  let neg = af::mul(&af::sub(&1.0, target, false).unwrap(), &af::log(&(af::sub(&1.0, pred, false).unwrap())).unwrap(), false).unwrap(); //[1-y]ln[1-x]
  let e = af::sub(&pos, &neg, false).unwrap();
  af::sum_all(&e).unwrap().0 as f32
}

pub fn mse_derivative(pred: &Array, target: &Array) -> Array {
  af::sub(pred, target, false).unwrap()
}

pub fn cross_entropy_derivative(pred: &Array, target: &Array) -> Array {
  mse_derivative(pred, target)
}

pub fn loss_delta(prediction: &Array, target: &Array
              , loss: &'static str, activation_type: &'static str) -> Array
{
  // d_L = d_loss * d(z) where z = activation w/out non-linearity (& in this case the predictions)
  let activated_prediction = activations::get_activation(activation_type, prediction).unwrap();
  let d_loss = get_loss_derivative(loss, &activated_prediction, target).unwrap();
  let d_z = activations::get_activation_derivative(activation_type, &activated_prediction).unwrap();
  af::mul(&d_loss, &d_z, false).unwrap()
}

pub fn get_loss(name: &str, pred: &Array, target: &Array) -> Result<f32, HALError> {
  match name {
    "mse"           => Ok(mse(pred, target)),
    "cross_entropy" => Ok(cross_entropy(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}

pub fn get_loss_derivative(name: &str, pred: &Array, target: &Array) -> Result<Array, HALError> {
  match name {
    "mse"           => Ok(mse_derivative(pred, target)),
    "cross_entropy" => Ok(cross_entropy_derivative(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}
