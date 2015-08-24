//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{Array};
use layer::Layer;
use optimizer::Optimizer;

pub trait Model {
  fn new(optimizer: Box<Optimizer>, loss: &'static str) -> Self;
  fn fit(&mut self, input: &Array, target: &Array
         , batch_size: u64, iter: u64
         , verbose: bool) -> (Vec<f32>, Array);
  fn forward(&mut self, activation: &Array) -> Array;
  fn backward(&mut self, prediction: &Array, target: &Array) -> (f32, Array);
  fn add(&mut self, layer: Box<Layer>);
  fn info(&self);
}
