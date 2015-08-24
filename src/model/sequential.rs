use af;
use af::{Array};
use std::default::Default;
use std::cell::Cell;
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
    //println!("loss : {}\noptimizer: {}", self.loss, self.optimizer);
    println!("loss : {}", self.loss);
  }

  fn forward(&mut self, activation: &Array) -> Array {
    let mut a = self.layers[0].forward(activation);
    for i in 1..self.layers.len() {
      a = self.layers[i].forward(&a);
    }
    a
  }

  fn fit(&mut self, input: &Array, target: &Array
         , batch_size: u64, iter: u64
         , verbose: bool) -> (Vec<f32>, Array)
  {
    let mut fwd_pass = target.copy().unwrap(); // sizing
    let mut lossvec = Vec::<f32>::new();
    
    for i in (0..iter) {
      //TODO: Minitbatch here
      fwd_pass = self.forward(input);
      let (l, _) = self.backward(&fwd_pass, target); 
      lossvec.push(l);

      if(verbose){
        println!("loss: {}", l);
      }
    }
    (lossvec, fwd_pass)
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> (f32, Array) {
    self.optimizer.update(&mut self.layers, prediction, target, self.loss)
  }
}
