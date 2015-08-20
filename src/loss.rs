use af;
use af::{Dim4, Array};
use error::HALError;

pub fn mse(x: &Array, y: &Array) -> Array {
  let diff = af::sum(y, af::mul(-1.0, x));
  af::div(af::sum(af::dot(diff, diff)), 2.0).unwrap()
}

pub fn cross_entropy(pred: &Array, target: &Array) -> Array {
  // y: true target, x: prediction
  // E = sum(-ylnx - [1-y]ln[1-x])
  af::sum(af::sub(af::mul(af::mul(-1.0, target), af::log(pred)) // -ylnx
                  , af::mul(af::sub(1.0, target), af::log(af::sub(1.0, pred))))).unwrap() //[1-y]ln[1-x]
}

pub fn mse_derivative(pred: &Array, target: &Array) -> Array {
  af::sub(pred, target).unwrap()
}

pub fn cross_entropy_derivative(pred: &Array, target: &Array) -> Array {
  mse_derivative(pred, target)
}

pub fn get_loss(name: &str, pred: &Array, target: &Array) -> Array {
  match(name){
    "mse"           => mse(pred, target),
    "cross_entropy" => cross_entropy(pred, target),
    _               => HALError::UNKNOWN,
  }
}

pub fn get_loss_derivative(name: &str, pred: &Array, target: &Array) -> Array {
  match(name){
    "mse"           => mse_derivative(pred, target),
    "cross_entropy" => cross_entropy_derivative(pred, target),
    _               => HALError::UNKNOWN,
  }
}
