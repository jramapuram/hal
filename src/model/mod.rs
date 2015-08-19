//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{Array};
use layer::Layer;
use optimizer::Optimizer;

pub trait Model {
  fn new(optimizer: Box<Optimizer>, loss: &'static str) -> Self;
  fn forward(&self, activation: &Array) -> Array;
  fn backward(&self, target: &Array, train: bool);
  fn add(&mut self, layer: Box<Layer>);
  fn info(&self);
}
