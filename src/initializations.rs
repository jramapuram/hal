use af;
use num::Complex;
use error::HALError;
use af::{Dim4, Array, HasAfEnum};

pub fn get_fans(dims: Dim4) -> (f32, f32){
  let ndims = dims.ndims();
  let fan_in = match ndims {
    2  => dims[0],
    _  => dims.get()[1..ndims].iter().fold(1, |prod, x| prod * x) as u64,
  };
  let fan_out = match dims[1] {
    2  => dims[1],
    _  => dims[0],
  };
  (fan_in as f32, fan_out as f32)
}

pub fn normal<T: HasAfEnum>(dims: Dim4, scale: f64) -> Array {
  af::mul(&af::randn::<T>(dims).unwrap(), &scale, false)
    .unwrap().cast::<T>().unwrap()
}

pub fn uniform<T: HasAfEnum>(dims: Dim4, scale: f32) -> Array{
  af::sub(&af::mul(&af::randu::<f32>(dims).unwrap(), &scale, false).unwrap()
          , &scale, false).unwrap().cast::<T>().unwrap()
}

pub fn zeros<T: HasAfEnum>(dims: Dim4) -> Array {
  af::constant(0.0 as f32, dims).unwrap().cast::<T>().unwrap()
}

pub fn ones<T: HasAfEnum>(dims: Dim4) -> Array {
  af::constant(1.0 as f32, dims).unwrap().cast::<T>().unwrap()
}

pub fn glorot_uniform<T: HasAfEnum>(dims: Dim4) -> Array {
  let (fan_in, fan_out) = get_fans(dims);
  let s = (6.0f32 / (fan_in + fan_out)).sqrt();
  uniform::<T>(dims, s)
}

pub fn glorot_normal<T: HasAfEnum>(dims: Dim4) -> Array {
  let (fan_in, fan_out) = get_fans(dims);
  let s = (2.0f32 / (fan_in + fan_out)).sqrt();
  normal::<T>(dims, s as f64)
}

pub fn lecun_uniform<T: HasAfEnum>(dims: Dim4) -> Array {
  let (fan_in, _) = get_fans(dims);
  let s = 3.0f32 / fan_in;
  uniform::<T>(dims, s)
}

//TODO: Orthogonal

pub fn get_initialization<T: HasAfEnum>(name: &str, dims: Dim4) -> Result<Array, HALError>
{
  match name {
    "glorot_uniform" => Ok(glorot_uniform::<T>(dims)),
    "glorot_normal"  => Ok(glorot_normal::<T>(dims)),
    "lecun_uniform"  => Ok(lecun_uniform::<T>(dims)),
    "normal"         => Ok(normal::<T>(dims, 0.05f64)), //TODO: Parameterize
    "uniform"        => Ok(uniform::<T>(dims, 0.05f32)), //TODO: Parameterize
    "zeros"          => Ok(zeros::<T>(dims)),
    "ones"           => Ok(ones::<T>(dims)),
    _                => Err(HALError::UNKNOWN),
  }
}
