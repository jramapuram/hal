use af;
use af::{Array, Dim4};
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
  pub clip_grad: u64,
  pub iter: u64,
  velocity_W: Vec<Array>,
  velocity_b: Vec<Array>,
}

impl Default for SGD {
  fn default() -> SGD {
    SGD {
      name: "SGD".to_string(),
      learning_rate: 0.01,
      momemtum: 0.1,
      decay: 0.0,
      nesterov: false,
      clip_grad: 0,
      iter: 0,
      velocity_W: Vec::new(),
      velocity_b: Vec::new(),
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
      clip_grad: params.get("clip_grad").unwrap().parse::<u64>().unwrap(),
      iter: 0,
      velocity_W: Vec::new(),
      velocity_b: Vec::new(),
    }
  }

  fn setup(&mut self, w_dims: Vec<Dim4>, b_dims: Vec<Dim4>){
    for w in w_dims {
      self.velocity_W.push(initializations::zeros(w));
    }

    for b in b_dims {
      self.velocity_b.push(initializations::zeros(b));
    }
  }

  fn update(&mut self, parameter_manager: &mut ParamManager, batch_size: u64)
  {
    self.iter += 1;
    self.learning_rate *= 1.0 / (1.0 + self.decay * (self.iter as f32));
    let alpha = self.learning_rate / batch_size as f32;
    let mut velocity_index = [0, 0];

    for layer_num in 0..parameter_manager.num_layers(){
      // d_w = delta * a_{t-1} / minibatch_size
      // v   = momemtum * v + learning_rate * d_w
      // W   = W - v
      let weights = parameter_manager.get_weights(layer_num);
      for weight_num in (0..weights.len()) {
        let delta = parameter_manager.get_delta(layer_num, weight_num);
        let activ = parameter_manager.get_input(layer_num, weight_num).data;
        let w_update = af::div(&af::matmul(&activ, &delta
                                           , af::MatProp::TRANS
                                           , af::MatProp::NONE).unwrap()
                               , &batch_size, false).unwrap(); // divide by the batch size
        self.velocity_W[velocity_index[0]] = af::mul(&self.momemtum, &self.velocity_W[velocity_index[0]], false).unwrap();
        self.velocity_W[velocity_index[0]] = af::sub(&self.velocity_W[velocity_index[0]]
                                                     , &af::mul(&alpha, &w_update, true).unwrap(), false).unwrap();
        parameter_manager.set_weight(layer_num, weight_num
                                     , af::add(&weights[weight_num], &self.velocity_W[velocity_index[0]], false).unwrap());
        velocity_index[0] += 1;
      }

      // d_b = sum(delta, 1) / minibatch_size
      // v = momemtum * v + learning_rate * d_b
      // b = b - v
      let biases = parameter_manager.get_biases(layer_num);
      for bias_num in (0..biases.len()) {
        let delta = parameter_manager.get_delta(layer_num, bias_num);
        let b_update = af::transpose(&af::div(&af::sum(&delta, 0).unwrap(), &batch_size, true).unwrap(), false).unwrap(); // divide by the batch size
        self.velocity_b[velocity_index[1]] = af::mul(&self.momemtum, &self.velocity_b[velocity_index[1]], false).unwrap();
        self.velocity_b[velocity_index[1]] = af::sub(&self.velocity_b[velocity_index[1]]
                                                     , &af::mul(&alpha, &b_update, false).unwrap(), true).unwrap();
        parameter_manager.set_bias(layer_num, bias_num
                                   , af::sub(&biases[bias_num], &self.velocity_b[velocity_index[1]], true).unwrap());
        velocity_index[1] += 1;
      }
    }
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
