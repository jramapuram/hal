//mod Optimizer;
//mod Loss;

use layer;
use std::collections::LinkedList;

trait Model{
  fn new(optimizer: String, loss: String) -> Self;
  //fn forward(&self, activation: &Array) -> Array;
  //fn backward(&self, inputs: &Array, gradients: &Array);
  fn info(&self);
}

pub struct Sequential {
  layers: LinkedList<Box<layer::Layer>>,
  optimizer: String,//Optimizer,
  loss: String,//Loss,
}

impl Model for Sequential {
  fn new(optimizer: String, loss: String) -> Sequential {
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
