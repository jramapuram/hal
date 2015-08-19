use layer::{Layer};
use model::Model;
use optimizer::{Optimizer, get_optimizer, get_default_optimizer}
use af::{Array, mul, sub};
use loss::{get_loss, get_loss_derivative};
use std::default::Default;

pub struct Sequential {
  layers: Vec<Box<Layer>>,
  optimizer: Box<Optimizer>,
  loss: &'static str,
}

// TODO: implement default trait
// impl Default for Sequential{
//   fn default(optimizer: &'static str, loss: &'static str) -> Sequential {
//     Sequential {
//       layers: Vec::new(),
//       loss: loss,
//       optimizer: get_default_optimizer(optimizer),
//     }
//   }  
// }

impl Model for Sequential {
  fn new(optimizer: Box<Optimizer>, loss: &'static str) -> Sequential {
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
    //TODO: convert to log crate
    println!("loss : {}\noptimizer: {}", self.loss, self.optimizer);
  }

  fn forward(&self, activation: &Array) -> Array {
    let mut a = self.layers[0].forward(activation);
    for i in 1..self.layers.len() {
      a = self.layers[i].forward(&a);
    }
    a
  }

  fn backward(&self, target: &Array, train: bool) {
    // d_L = d_loss * d(z) where z = activation w/out non-linearity
    let prediction = self.layers.last().unwrap().get_activation();
    let d_loss = get_loss_derivative(self.loss, prediction, target);
    let d_z = get_activation_derivative(self.layers.last().get_input());
    let mut diffs = self.layers.last().set_diffs(mul(d_loss, d_z));

    for i in (0..self.layers.len() - 1).rev() {
      diffs = self.layers[i].backward(&diffs
                                      , self.optimizer.grads(self.layers[i].get_activation())
                                      , train);
    }
  }
}
