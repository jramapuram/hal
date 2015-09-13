use af;
use af::Array;
use error::HALError;

pub fn mse(pred: &Array, target: &Array) -> f32 {
  let diff = af::sub(pred, target).unwrap();
  (af::sum_all(&af::mul(&diff, &diff).unwrap()).unwrap().0/2.0) as f32
}

pub fn cross_entropy(pred: &Array, target: &Array) -> f32 {
  // y: true target, x: prediction
  // E = sum(-ylnx - [1-y]ln[1-x])
  let pos = af::mul(&af::mul(&-1.0, target).unwrap(), &af::log(&pred).unwrap()).unwrap(); // -ylnx
  let neg = af::mul(&af::sub(&1.0, target).unwrap(), &af::log(&(af::sub(&1.0, pred).unwrap())).unwrap()).unwrap(); //[1-y]ln[1-x]
  let e = af::sub(&pos, &neg).unwrap();
  af::sum_all(&e).unwrap().0 as f32
}

pub fn mse_derivative(pred: &Array, target: &Array) -> Array {
  af::sub(pred, target).unwrap()
}

pub fn cross_entropy_derivative(pred: &Array, target: &Array) -> Array {
  mse_derivative(pred, target)
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
