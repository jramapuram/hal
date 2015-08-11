pub use self::sgd::SGD;
mod sgd;

use af::{Array};
use layer::Layer;
use std::collections::HashMap;

pub trait Optimizer {
  fn new(params: &HashMap) -> Self;
  fn grads(layer: &Layer) -> Array;
  fn update(layer: &mut Layer, grads: &Array);
  fn info();
}
