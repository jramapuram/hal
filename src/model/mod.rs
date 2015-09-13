//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{Array};
use na::DMat;
use layer::Layer;
use optimizer::Optimizer;

pub trait Model {
  fn new(optimizer: Box<Optimizer>, loss: &'static str) -> Self;
  fn fit(&mut self, input: &mut DMat<f32>, target: &mut DMat<f32>
         , batch_size: u64, iter: u64
         , shuffle: bool, verbose: bool) -> (Vec<f32>, DMat<f32>);
  fn forward(&mut self, activation: &Array) -> Array;
  fn backward(&mut self, prediction: &Array, target: &Array) -> f32;
  fn add(&mut self, layer: Box<Layer>);
  fn set_device(&mut self, device_id: i32);
  fn info(&self);
}
