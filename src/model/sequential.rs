use af;
use af::{Array};
use std::default::Default;

use layer::{Layer};
use model::Model;
use optimizer::Optimizer;

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

  fn fit(&self, input: &Array, target: &Array
         , batch_size: u64, iter: u64
         , verbose: bool) -> (Vec<Array>, Array)
  {
    let mut fwd_pass = target.copy(); // sizing
    let mut loss = Vec::new();
    
    for i in (0..iter) {
      //TODO: Minitbatch here
      fwd_pass = self.forward(input);
      loss.push(self.backward(fwd_pass, target));

      if(verbose){
        println!("loss:");
        af::print(loss.last());
      }
    }
    (loss, fwd_pass)
  }

  fn backward(&self, prediction: &Array, target: &Array) -> (Vec<Array>, Array) {
    self.optimizer.update(self.layers, prediction, target, self.loss);
  }
}
