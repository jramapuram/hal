//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{Array};
use layer::Layer;

pub trait Model {
  fn new(optimizer: &'static str, loss: &'static str) -> Self;
  fn forward(&self, activation: &Array) -> Array;
  fn backward(&self, target: &Array);
  fn add(&mut self, layer: Box<Layer>);
  fn info(&self);
}
