use af;
use af::{Array, Dim4};
use na::{DMat, Shape, Transpose};
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
  device: i32,
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
      device: -1,
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

  //TODO: Models in parallel on different GPU's 
  fn set_device(&mut self, device_id: i32){
    self.device = device_id;
    af::set_device(device_id);
    af::info();
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
    println!("\ntrain samples: {:?} | target samples: {:?} | batch size: {}"
             , input.shape(), target.shape(), batch_size);
    self.optimizer.setup(&self.layers);
      
    // create the container to hold the forward pass & loss results
    let mut forward_pass = initializations::zeros(Dim4::new(&[1, input.ncols() as u64, 1, 1]));
    let mut lossvec = Vec::<f32>::new();
    let mut loss: f32 = 0.0f32;

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
        print!("\n[iter: {}] ", i);
      }

      // over every batch
      let incols = input.ncols();
      let tncols = target.ncols();
      for (i, t) in Zip::new((input.transpose().as_vec().chunks(incols * batch_size as usize)
                              , target.transpose().as_vec().chunks(tncols * batch_size as usize)))
      {
        // column major order is preferred for BLAS
        let batch_input = utils::normalize(&utils::raw_to_array(i, incols,  batch_size as usize), 3.0f32);
        let batch_target = utils::normalize(&utils::raw_to_array(t, tncols, batch_size as usize), 3.0f32);
        
        // println!("batched [input: {:?} | target: {:?}]"
        //          , batch_input.dims().unwrap()
        //          , batch_target.dims().unwrap());

        for row_num in 0..batch_size { //over every row in batch
          // af::print(&af::col(&batch_input, row_num).unwrap());
          forward_pass = self.forward(&af::col(&batch_input, row_num).unwrap());
          loss = self.backward(&forward_pass, &af::col(&batch_target, row_num).unwrap());
        }

        self.optimizer.update_parameters(&mut self.layers, batch_size);
      }

      lossvec.push(loss);
      if verbose {
        print!("{} ", loss);
      }
    }

    utils::write_csv::<f32>("output.csv", &lossvec);
    (lossvec, utils::array_to_dmat(&forward_pass))
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> f32 {
    self.optimizer.optimize(&mut self.layers, prediction, target, self.loss)
  }
}
