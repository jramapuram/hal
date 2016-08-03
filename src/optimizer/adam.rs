use af;
use af::{Array, Dim4};
use itertools::Zip;
use std::collections::HashMap;
use std::default::Default;

use optimizer;
use params::ParamManager;
use initializations;
use optimizer::Optimizer;

#[allow(non_snake_case)]
pub struct Adam {
  pub name: String,
  pub learning_rate: f32,
  pub beta1: f32,
  pub beta2: f32,
  pub eps: f32,
  pub lambda: f32,
  pub clip_grad: f32,
  pub iter: u64,
  mt: Vec<Array>,
  vt: Vec<Array>,
}

impl Default for Adam {
  fn default() -> Adam {
    Adam {
      name: "Adam".to_string(),
      learning_rate: 1e-3,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      lambda: 1.0 - 1e-8,
      clip_grad: 5.0,
      iter: 0,
      mt: Vec::new(),
      vt: Vec::new(),
    }
  }
}

impl Optimizer for Adam {
  fn new(params: &HashMap<&str, &str>) -> Adam {
    Adam{
      name: "Adam".to_string(),
      learning_rate: params.get("learning_rate").unwrap().parse::<f32>().unwrap(),
      beta1: params.get("beta1").unwrap().parse::<f32>().unwrap(),
      beta2: params.get("beta2").unwrap().parse::<f32>().unwrap(),
      eps: params.get("eps").unwrap().parse::<f32>().unwrap(),
      lambda: params.get("lambda").unwrap().parse::<f32>().unwrap(),
      clip_grad: params.get("clip_grad").unwrap().parse::<f32>().unwrap(),
      iter: 0,
      mt: Vec::new(),
      vt: Vec::new(),
    }
  }

  fn setup(&mut self, dims: Vec<Dim4>) {
    if self.mt.len() == 0 {
      for dim in dims {
        self.vt.push(initializations::zeros::<f32>(dim));
        self.mt.push(initializations::zeros::<f32>(dim));
      }
    }
  }

  fn update(&mut self, parameter_manager: &mut ParamManager, batch_size: u64)
  {
    self.iter += 1;
    // let lr = self.learning_rate * (1.0 / (1.0 + self.decay * (self.iter as f32)));
    // let alpha = lr / batch_size as f32;
    self.beta1 = self.beta1 * self.lambda;

    // all arrays are returned as [W0, b0, .. WN, bN, ..] (note this is per layer)
    // deltas are returned in the same way
    let num_params = self.vt.len();
    for (arr, delta, vt_i, mt_i, ind) in Zip::new((parameter_manager.get_all_arrays().iter()   // weights + biases
                                                   , parameter_manager.get_all_deltas().iter() // deltas of above
                                                   , self.vt.iter_mut()                        // vt
                                                   , self.mt.iter_mut()                        // mt
                                                   , 0..num_params))                           // current index
    {
      let grad_update = match self.clip_grad > 0.0 {
        false => delta.clone(),
        true  => optimizer::clip_grads(&delta, self.clip_grad),
      };

      *mt_i = af::add(&af::mul(&self.beta1, mt_i, false)
                      , &af::mul(&(1.0 - self.beta1), &grad_update, false)
                      , false);
      *vt_i = af::add(&af::mul(&self.beta2, vt_i, false)
                      , &af::mul(&(1.0 - self.beta2), &af::mul(&grad_update, &grad_update, false), false)
                      , false);
      let mhat_i = af::div(mt_i, &(1.0 - self.beta1), false);
      let vhat_i = af::div(vt_i, &(1.0 - self.beta2), false);
      let update = af::mul(&self.learning_rate, &af::div(&mhat_i, &af::add(&af::sqrt(&vhat_i), &self.eps, false), false)
                           , false);

      parameter_manager.set_array_from_index(af::sub(arr, &update, false), ind);
    }

    // zero out the deltas
    parameter_manager.zero_all_deltas();
    parameter_manager.zero_all_state_derivatives();
  }

  fn info(&self){
    println!("optimizer_name: {}", self.name);
    println!("learning_rate:  {}", self.learning_rate);
    println!("beta1:          {}", self.beta1);
    println!("beta2:          {}", self.beta2);
    println!("eps:            {}", self.eps);
    println!("lambda:         {}", self.lambda);
    println!("clip_grad:      {}", self.clip_grad);
    println!("iter:           {}", self.iter);
  }
}
