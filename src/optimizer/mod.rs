pub use self::sgd::SGD;
mod sgd;

use std::collections::HashMap;

use layer::Layer;
use error::HALError;

pub trait Optimizer {
  fn new(params: &HashMap<&str, &str>) -> Self where Self: Sized;
  fn setup(&mut self, layers: &Vec<Box<Layer>>);
  fn update(&mut self, layers: &mut Vec<Box<Layer>>, batch_size: u64);
  fn info(&self);
}

pub fn get_optimizer(name: &str, params: &HashMap<&str, &str>) -> Result<Box<Optimizer>, HALError>{
  match name{
    "sgd" => Ok(Box::new(SGD::new(params))),
    _     => Err(HALError::UNKNOWN),
  }
}

pub fn get_default_optimizer(name: &str) -> Result<Box<Optimizer>, HALError>{
  match name{
    "sgd" => Ok(Box::new(SGD::default())),
    _     => Err(HALError::UNKNOWN),
  }
}
