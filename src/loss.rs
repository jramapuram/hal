use af::Dim4;
use af::Array;
use error::HALError;

pub fn mse(x: &Array, y: &Array) -> Array {
  let diff = af::sum(y, af::mul(-1.0, x));
  af::div(af::sum(af::dot(diff, diff)), 2.0)
}

pub fn cross_entropy(x: &Array, y: &Array) -> Array {
  af::sum(af::mul(y, af::log(x)))
}

pub fn mse_derivative(x: &Array, y: &Array) -> Array {
  af::sub(x, y)
}

pub fn cross_entropy_derivative(x: &Array) -> Array {
  af::sub(x, y)
}

pub fn get_loss(name: &str, x: &Array) -> Array {
  match(name){
    "mse"           => mse(x),
    "cross_entropy" => cross_entropy(x),
    _               => HALError::UNKNOWN,
  }
}

pub fn get_loss_derivative(name: &str, x: &Array) -> Array {
  match(name){
    "mse"           => mse_derivative(x),
    "cross_entropy" => cross_entropy_derivative(x),
    _               => HALError::UNKNOWN,
  }
}
