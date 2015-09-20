use af;
use af::{Array, Dim4};
use na::{DMat, Shape, Transpose};
use std::default::Default;
use itertools::Zip;

use utils;
use loss;
use initializations;
use layer::{Layer, Input};
use model::Model;
use optimizer::{Optimizer, SGD};

pub struct Sequential {
  layers: Vec<Box<Layer>>,
  optimizer: Box<Optimizer>,
  loss: &'static str,
  device: i32,
}

impl Default for Sequential{
  fn default() -> Sequential {
    Sequential {
      layers: Vec::new(),
      optimizer: Box::new(SGD::default()),
      loss: "mse",
      device: 0,
    }
  }
}

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

  //TODO: convert to log crate
  fn info(&self) {
    match af::info() {
      Ok(_)   => {},
      Err(e)  => panic!("could not get info: {:?}", e),
    };

    println!("");
    self.optimizer.info();
    println!("loss:           {}\nnum_layers:     {}", self.loss, self.layers.len());
  }

  fn set_device(&mut self, device_id: i32) {
    self.device = device_id;
    match af::set_device(device_id) {
      Ok(_)  => {},
      Err(e) =>  panic!("could not set device: {:?}", e),
     };
  }

  fn forward(&mut self, activation: &Array) -> Array {
    let mut activate = Input {data: vec![activation.clone()], activation: vec!["ones"]};
    for i in 0..self.layers.len() {
      activate = self.layers[i].forward(&activate);
    }
    activate.data.last().unwrap().clone() //NOTE: This is non-activated output
  }

  fn fit(&mut self, input: &mut DMat<f32>, target: &mut DMat<f32>
         , batch_size: u64, shuffle: bool, verbose: bool) -> (Vec<f32>, DMat<f32>)
  {
    // some required data validity checks
    let iter = input.nrows() as u64 / batch_size;
    println!("\ntrain samples: {:?} | target samples: {:?} | batch size: {} | iterations: {}"
             , input.shape(), target.shape(), batch_size, iter);
    assert!(target.nrows() == input.nrows());
    assert!(input.nrows() as u64 >= batch_size
            && input.nrows() as u64 % batch_size == 0);
    self.optimizer.setup(&self.layers);

    // create the container to hold the forward pass & loss results
    let mut forward_pass = initializations::zeros(Dim4::new(&[1, input.ncols() as u64, 1, 1]));
    let mut lossvec = Vec::<f32>::new();
    let mut loss: f32;// = 0.0f32;

    // randomly shuffle the data
    if shuffle {
      let col_vec = [input.ncols(), target.ncols()];
      utils::shuffle(&mut[input.as_mut_vec(), target.as_mut_vec()], &col_vec, false);
    }

    // normalize the data by mean and 3 std deviations
    *input = utils::normalize_dmat(input, 3.0f32);
    *target = utils::normalize_dmat(target, 3.0f32);

    // over every batch
    let incols = input.ncols();
    let tncols = target.ncols();
    let mut current_iteration = 0;
    for (i, t) in Zip::new((input.transpose().as_vec().chunks(incols * batch_size as usize)
                            , target.transpose().as_vec().chunks(tncols * batch_size as usize)))
    {
      if verbose {
        print!("\n[iter: {}] ", current_iteration);
        current_iteration += 1;
      }

      // column major order is preferred for BLAS
      let batch_input  = utils::raw_to_array(i, incols,  batch_size as usize);
      let batch_target = utils::raw_to_array(t, tncols, batch_size as usize);

      // DEBUG:
      // println!("batched [input: {:?} | target: {:?}]"
      //          , batch_input.dims().unwrap()
      //          , batch_target.dims().unwrap());

      forward_pass = self.forward(&batch_input);
      loss = self.backward(&forward_pass, &batch_target);
      self.optimizer.update(&mut self.layers, batch_size);

      lossvec.push(loss);
      if verbose {
        print!("{} ", loss);
      }
    }

    utils::write_csv::<f32>("loss.csv", &lossvec);
    (lossvec, utils::array_to_dmat(&forward_pass))
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> f32 {
    //    self.optimizer.optimize(&mut self.layers, prediction, target, self.loss)
    let last_index = self.layers.len() - 1;
    let mut delta = loss::loss_delta(prediction
                                     , target
                                     , self.loss
                                     , self.layers[last_index].get_activation_type());
    for i in (0..last_index + 1).rev() {
      delta = self.layers[i].backward(&delta);
    }

    loss::get_loss(self.loss, prediction, target).unwrap()
  }
}
