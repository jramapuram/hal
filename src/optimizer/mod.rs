pub use self::sgd::SGD;
mod sgd;

use std::collections::HashMap;

use af::{Array};
use layer::Layer;
use error::HALError;

pub trait Optimizer {
  fn new(params: &HashMap<&str, &str>) -> Self where Self: Sized;
  fn grads(&self, prediction: &Array, target: &Array, input: &Array
           , loss: &'static str, activation_type: &'static str) -> Array;
  fn update(&self, layers: &mut Vec<Layer>
            , prediction: &Array
            , target: &Array
            , loss: &'static str) -> (Array, Array);
  fn update_one(&self, layer: &mut Layer, prev_activation: &Array, diffs: &Array);
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
