use af;
use af::{Dim4, Array, HasAfEnum};

use utils;
use error::HALError;

/// A helper to provide the scaling for uniform and normal
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

/// A helper to return a normal shape with the provided scale
pub fn normal<T: HasAfEnum>(dims: Dim4, scale: f32) -> Array {
  let src_type = T::get_af_dtype();
  let scale_vec = utils::constant(dims, src_type, scale);
  let u = af::mul(&af::randn::<T>(dims).unwrap(), &scale_vec, false).unwrap();
  let dst_type = u.get_type().unwrap();
  assert!(src_type == dst_type
          , "type mismatch detected in normal, {:?} vs {:?}"
          , src_type, dst_type);
  u
}

/// A helper to provide a uniform shape with the provided scale
pub fn uniform<T: HasAfEnum>(dims: Dim4, scale: f32) -> Array{
  let src_type = T::get_af_dtype();
  let scale_vec = utils::constant(dims, src_type, scale);
  let u = af::sub(&af::mul(&af::randu::<T>(dims).unwrap(), &scale_vec, false).unwrap()
                  , &scale, false).unwrap();
  let dst_type = u.get_type().unwrap();
  assert!(src_type == dst_type
          , "type mismatch detected in uniform, {:?} vs {:?}"
          , src_type, dst_type);
  u
}

/// A helper to provide a shape of zeros
pub fn zeros<T: HasAfEnum>(dims: Dim4) -> Array {
  utils::constant(dims, T::get_af_dtype(), 0.0f32)
}

/// A helper to provide a shape of ones
pub fn ones<T: HasAfEnum>(dims: Dim4) -> Array {
  utils::constant(dims, T::get_af_dtype(), 1.0f32)
}

/// A helper to provide a shape of glorot uniform initialized values
pub fn glorot_uniform<T: HasAfEnum>(dims: Dim4) -> Array {
  let (fan_in, fan_out) = get_fans(dims);
  let s = (6.0f32 / (fan_in + fan_out)).sqrt();
  uniform::<T>(dims, s)
}

/// A helper to provide a shape of glorot normal initialized values
pub fn glorot_normal<T: HasAfEnum>(dims: Dim4) -> Array {
  let (fan_in, fan_out) = get_fans(dims);
  let s = (2.0f32 / (fan_in + fan_out)).sqrt();
  normal::<T>(dims, s)
}

/// A helper to provide a shape of lecun uniform initialized values
pub fn lecun_uniform<T: HasAfEnum>(dims: Dim4) -> Array {
  let (fan_in, _) = get_fans(dims);
  let s = 3.0f32 / fan_in;
  uniform::<T>(dims, s)
}

//TODO: permut
pub fn permut<T: HasAfEnum>(dims: Dim4) -> Array{
    af::range::<T>(dims, 0).unwrap()
}
//TODO: Orthogonal

/// A helper to retrieve an initialization based on a name and a shape
pub fn get_initialization<T: HasAfEnum>(name: &str, dims: Dim4) -> Result<Array, HALError>
{
  match name {
    "glorot_uniform" => Ok(glorot_uniform::<T>(dims)),
    "glorot_normal"  => Ok(glorot_normal::<T>(dims)),
    "lecun_uniform"  => Ok(lecun_uniform::<T>(dims)),
    "normal"         => Ok(normal::<T>(dims, 0.05f32)),  //TODO: Parameterize
    "uniform"        => Ok(uniform::<T>(dims, 0.05f32)), //TODO: Parameterize
    "zeros"          => Ok(zeros::<T>(dims)),
    "ones"           => Ok(ones::<T>(dims)),
    "permut"         => Ok(permut::<T>(dims)),
    _                => Err(HALError::UNKNOWN),
  }
}
