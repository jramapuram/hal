use af;
use af::{Array, Dim4};
use na::{DMat, Shape};
//use std::default::Default;
use itertools::Zip;

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
         , shuffle: bool, verbose: bool) -> (Vec<f32>, DMat<f32>)
  {
    println!("train samples: {:?} | target samples: {:?} | batch size: {}"
             , input.shape(), target.shape(), batch_size);
    //let output_dims = self.layers[self.layers.len() - 1].output_size();

    // create the container to hold the forward pass & loss results
    let dims = Dim4::new(&[1, input.ncols() as u64, 1, 1]);
    let mut forward_pass = initializations::zeros(dims);
    let mut lossvec = Vec::<f32>::new();

    // randomly shuffle the data
    assert!(target.nrows() == input.nrows());
    assert!(input.nrows() as u64 >= batch_size
            && input.nrows() as u64 % batch_size == 0);
    if shuffle {
      let col_vec = [input.ncols(), target.ncols()];
      utils::shuffle(&mut[input.as_mut_vec(), target.as_mut_vec()], &col_vec, false);
    }
    
    for i in (0..iter) { // over number of iterations
      if verbose {
        println!("iter: {}", i);
      }
 
      // over every batch
      for (i, t) in Zip::new((input.as_vec().chunks(batch_size as usize)
                              , target.as_vec().chunks(batch_size as usize)))
      {
        let batch_input = utils::normalize(&utils::raw_to_array(i, batch_size as usize, input.ncols()), 3.0f32);
        let batch_target = utils::normalize(&utils::raw_to_array(t, batch_size as usize, target.ncols()), 3.0f32);

        for row_num in 0..batch_size { //over every row in batch
          forward_pass = self.forward(&af::transpose(&af::row(&batch_input, row_num).unwrap(), false).unwrap());
          let l = self.backward(&forward_pass, &af::transpose(&af::row(&batch_target, row_num).unwrap(), false).unwrap());
          lossvec.push(l);
        }

        //TODO: Go here or below batch?
        if verbose {
          let last = lossvec.len();
          println!("loss: {}", lossvec[last - 1]);
        }
        self.optimizer.update_parameters(&mut self.layers);
      }
    }

    //let retval: DMat<f32> = 
    (lossvec, utils::array_to_dmat(&forward_pass))
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> f32 {
    self.optimizer.optimize(&mut self.layers, prediction, target, self.loss)
  }
}
