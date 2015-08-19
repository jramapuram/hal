pub use self::sgd::SGD;
mod sgd;

use std::collections::HashMap;

use af::{Array};
use layer::Layer;
use error::HALError;

pub trait Optimizer {
  fn new(&self, params: &HashMap<&str, &str>) -> Self;
  fn grads(&self, layer: &Layer) -> Array;
  fn update(&mut self, layer: &mut Layer, grads: &Array);
  fn info(&self);
}

pub fn get_optimizer(name: &str, params: &HashMap<&str, &str>){
  match name{
    "sgd" => SGD::new(params),
    _     => HALError::UNKNOWN,
  }
}

pub fn get_default_optimizer(name: &str){
  match name{
    "sgd" => SGD::default(),
    _     => HALError::UNKNOWN,
  }
}
