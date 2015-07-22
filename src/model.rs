//mod Optimizer;
//mod Loss;

use layer;
use std::collections::LinkedList;

pub trait Model {
  fn new(optimizer: &'static str, loss: &'static str) -> Self;
  //fn forward(&self, activation: &Array) -> Array;
  //fn backward(&self, inputs: &Array, gradients: &Array);
  fn info(&self);
}

pub struct Sequential {
  layers: LinkedList<Box<layer::Layer>>,
  optimizer: &'static str,
  loss: &'static str,
}

impl Model for Sequential {
  fn new(optimizer: &'static str, loss: &'static str) -> Sequential {
    Sequential {
      layers: LinkedList::new(),
      loss: loss,
      optimizer: optimizer,
    }
  }

  fn info(&self) {
    println!("loss : {}\noptimizer: {}", self.loss, self.optimizer);
  }
}
