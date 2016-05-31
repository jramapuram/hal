use af;
use af::{Array, Dim4, DType};
use itertools::Zip;
use std::collections::HashMap;
use std::default::Default;

use params::ParamManager;
use initializations;
use optimizer::Optimizer;

#[allow(non_snake_case)]
pub struct SGD {
  pub name: String,
  pub learning_rate: f32,
  pub momemtum: f32,
  pub decay: f32,
  pub nesterov: bool,
  pub clip_grad: f32,
  pub iter: u64,
  velocity: Vec<Array>,
}

impl Default for SGD {
  fn default() -> SGD {
    SGD {
      name: "SGD".to_string(),
      learning_rate: 1e-3,
      momemtum: 0.0,
      decay: 0.0,
      nesterov: false,
      clip_grad: 0.0,
      iter: 0,
      velocity: Vec::new(),
    }
  }
}

impl Optimizer for SGD {
  fn new(params: &HashMap<&str, &str>) -> SGD {
    SGD{
      name: "SGD".to_string(),
      learning_rate: params.get("learning_rate").unwrap().parse::<f32>().unwrap(),
      momemtum: params.get("momemtum").unwrap().parse::<f32>().unwrap(),
      decay: params.get("decay").unwrap().parse::<f32>().unwrap(),
      nesterov: params.get("nesterov").unwrap().parse::<bool>().unwrap(),
      clip_grad: params.get("clip_grad").unwrap().parse::<f32>().unwrap(),
      iter: 0,
      velocity: Vec::new(),
    }
  }

  fn setup(&mut self, dims: Vec<Dim4>) {  //, w_dims: Vec<Dim4>, b_dims: Vec<Dim4>){
    if self.velocity.len() == 0 {
      for dim in dims {
        self.velocity.push(initializations::zeros::<f32>(dim));
      }
    }
  }

  fn update(&mut self, parameter_manager: &mut ParamManager, batch_size: u64)
  {
    self.iter += 1;
    let lr = self.learning_rate * (1.0 / (1.0 + self.decay * (self.iter as f32)));
    let alpha = lr / batch_size as f32;
    let mut running_type = DType::F32;

    // all arrays are returned as [W0, b0, .. WN, bN, ..] (note this is per layer)
    // deltas are returned in the same way
    let num_params = self.velocity.len();
    for (arr, delta, velocity, ind) in Zip::new((parameter_manager.get_all_arrays().iter()   // weights + biases
                                                 , parameter_manager.get_all_deltas().iter() // deltas of above
                                                 , self.velocity.iter_mut()                  // velocity of above
                                                 , 0..num_params))                           // current index
    {
      // v   = momemtum * v + learning_rate * d_w (or d_b)
      // p   = p - v
      *velocity = af::add(&af::mul(&self.momemtum, velocity, false),
                          &af::mul(&alpha, delta, false), false);
      assert!(velocity.dims().get() == arr.dims().get());
      running_type = arr.get_type();
      parameter_manager.set_array_from_index(af::sub(arr, velocity, false), ind);
    }

    // zero out the deltas
    parameter_manager.zero_all_deltas(running_type);
  }

  fn info(&self){
    println!("optimizer_name: {}", self.name);
    println!("learning_rate:  {}", self.learning_rate);
    println!("momemtum:       {}", self.momemtum);
    println!("decay:          {}", self.decay);
    println!("nesterov:       {}", self.nesterov);
    println!("clip_grad:      {}", self.clip_grad);
  }
}
