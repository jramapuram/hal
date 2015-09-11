use af;
use af::{Array};
use na::DMat;
use std::default::Default;
use std::iter::Zip;

use utils;
use initializations;
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

  fn fit(&mut self, input: &mut DMat<f32>, target: &mut DMat<f32>
         , batch_size: u64, iter: u64
         , shuffle: bool, verbose: bool) -> (DMat<f32>, Array)
  {
    println!("train samples: {} | target samples: {} | batch size: {}"
             , input.len(), target.len(), batch_size);
    //let output_dims = self.layers[self.layers.len() - 1].output_size();

    // create the container to hold the forward pass & loss results
    let dims = Dim4::new(&[1, input.shape().1, 1, 1]);
    let mut forward_pass = initializations::zeros(dims);
    let mut lossvec = Vec::<f32>::new();

    // randomly shuffle the data
    assert!(target.nrows() == input.nrows());
    assert!(filtered_input.nrows() >= batch_size
            && filtered_input.nrows() % batch_size == 0);
    if shuffle {
      utils::shuffle(&mut[input.as_mut_vec(), target.as_mut_vec()]);
    }
    
    for i in (0..iter) {
      if verbose {
        println!("iter: {}", i);
      }
 
      //for row in (0..input.len().step_by(batch_size)) {
      for (i, t) in Zip::new((utils::batch(input, batch_size), utils::batch(target, batch_size)))
      {
        let batch_input = utils::raw_to_array(i);
        let batch_target = utils::raw_to_array(t);
        for row_num in 0..batch_input.dims()[0] {
          forward_pass = self.forward(&af::row(batch_input, row_num).unwrap());
          let (l, _) = self.backward(&forward_pass, &af::row(batch_target, row_num).unwrap()); 
          lossvec.push(l);

          if verbose{
            println!("loss: {}", l);
          }
        }
      }

      self.optimizer.update_parameters(&mut self.layers);
    }
    
    (lossvec, utils::array_to_dmat(forward_pass))
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> f32 {
    self.optimizer.optimize(&mut self.layers, prediction, target, self.loss)
  }
}
