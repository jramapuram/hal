//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{Array, AfBackend};
use std::collections::HashMap;

use optimizer::Optimizer;

pub trait Model {
  fn new(optimizer: Box<Optimizer>
         , loss: &str
         , backend: AfBackend
         , device: i32) -> Self;
  fn fit(&mut self, input: &mut Array, target: &mut Array
         , batch_size: u64, return_predictions: bool
         , shuffle: bool, verbose: bool) -> (Vec<f32>, Option<Vec<Array>>);
  fn forward(&mut self, activation: &Array, train: bool) -> Array;
  fn backward(&mut self, prediction: &Array, target: &Array) -> f32;
  fn add(&mut self, layer: &str
         , params: HashMap<&str, &str>);
  fn set_device(&mut self, backend: AfBackend, device_id: i32);
  fn info(&self);
}
