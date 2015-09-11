use af;
use error::HALError;
use af::{Dim4, Array};

pub fn normal(dims: Dim4) -> Array {
  af::randn(dims, af::Aftype::F32).unwrap()
}

pub fn uniform(dims: Dim4, spread: f32) -> Array{
  af::randu(dims, af::Aftype::F32).unwrap()//.and_then(|x| x * spread - spread / 2).unwrap()
}

pub fn zeros(dims: Dim4) -> Array {
  af::constant(0.0 as f32, dims).unwrap()
}

pub fn ones(dims: Dim4) -> Array {
  af::constant(1.0 as f32, dims).unwrap()
}

pub fn get_initialization(name: &str, dims: Dim4) -> Result<Array, HALError> {
  match name {
    "normal"  => Ok(normal(dims)),
    "uniform" => Ok(uniform(dims, 0.05)),
    "zeros"   => Ok(zeros(dims)),
    "ones"    => Ok(ones(dims)),
    _         => Err(HALError::UNKNOWN),
  }
}
