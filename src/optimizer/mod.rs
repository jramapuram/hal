pub use self::sgd::SGD;
mod sgd;

use af;
use af::{Array};
use std::collections::HashMap;

use loss;
use activations;
use layer::Layer;
use error::HALError;

pub trait Optimizer {
  fn new(params: &HashMap<&str, &str>) -> Self where Self: Sized;
  fn optimize(&mut self, layers: &mut Vec<Box<Layer>>
              , prediction: &Array
              , target: &Array
              , loss: &'static str) -> f32;
  fn update_parameters(&self, layers: &mut Vec<Box<Layer>>, batch_size: u64);
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

fn grads(prediction: &Array, target: &Array
         , loss: &'static str, activation_type: &'static str) -> Array
{
  // d_L = d_loss * d(z) where z = activation w/out non-linearity
  let d_loss = loss::get_loss_derivative(loss, prediction, target).unwrap();
  let d_z = activations::get_activation_derivative(activation_type, prediction).unwrap();
  af::mul(&d_loss, &d_z).unwrap()
}

fn update_delta(layer: &mut Box<Layer>, prev_activation: &Array, diffs: &Array)
{
  // delta = ( a_{l-1} * d_l, d_l )
  layer.update(&af::matmul(diffs, prev_activation, af::MatProp::NONE, af::MatProp::TRANS).unwrap(), diffs);
}
