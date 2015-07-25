use layer::{Layer};
use model::Model;
use af::{Array};

pub struct Sequential {
  layers: Vec<Box<Layer>>,
  optimizer: &'static str,
  loss: &'static str,
}

impl Model for Sequential {
  fn new(optimizer: &'static str, loss: &'static str) -> Sequential {
    Sequential {
      layers: Vec::new(),
      loss: loss,
      optimizer: optimizer,
    }
  }

  fn add(&mut self, layer: Box<Layer>) {
    self.layers.push(layer);
  }

  fn info(&self) {
    println!("loss : {}\noptimizer: {}", self.loss, self.optimizer);
  }

  fn forward(&self, activation: &Array) -> Array{
    let mut a = self.layers[0].forward(activation);
    for i in 1..self.layers.len() {
      a = self.layers[i].forward(&a);
    }
    a
  }
}
