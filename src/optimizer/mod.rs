pub use self::sgd::SGD;
mod sgd;

use std::collections::HashMap;

use af::{Array};
use layer::Layer;
use error::HALError;

pub trait Optimizer {
  fn new(params: &HashMap<&str, &str>) -> Self where Self: Sized;
  fn grads(&self, prediction: &Array, target: &Array
           , loss: &'static str, activation_type: &'static str) -> Array;
  fn optimize(&mut self, layers: &mut Vec<Box<Layer>>
              , prediction: &Array
              , target: &Array
              , loss: &'static str) -> f32;
  fn update_delta(&self, layer: &mut Box<Layer>, prev_activation: &Array, diffs: &Array);
  fn update_parameters(&self, layers: &mut Vec<Box<Layer>>);
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
