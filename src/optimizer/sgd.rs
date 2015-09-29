use af;
use af::{Array};
use std::collections::HashMap;
use std::default::Default;

use initializations;
use layer::{Layer};
use optimizer::Optimizer;

#[allow(non_snake_case)]
pub struct SGD {
  pub name: &'static str,
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
      name: "SGD",
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
      name: "SGD",
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

  fn setup(&mut self, layers: &Vec<Box<Layer>>){
    for layer in layers {
      for wdim in layer.get_weight_dims() {
        self.velocity_W.push(initializations::zeros(wdim));
      }
      for bdim in layer.get_bias_dims() {
        self.velocity_b.push(initializations::zeros(bdim));
      }
    }
  }

  fn update(&mut self, layers: &mut Vec<Box<Layer>>, batch_size: u64)
  {
    self.iter += 1;
    self.learning_rate *= 1.0 / (1.0 + self.decay * (self.iter as f32));
    let alpha = self.learning_rate / batch_size as f32;
    let mut velocity_index = [0, 0];

    for layer_num in 0..layers.len() {
      // v = momemtum * v + learning_rate * d_w
      // W = W - v
      let weights = layers[layer_num].get_weights();
      for weight_num in (0..weights.len()) {
        //DEBUG:
        // println!("weight size: {:?} | delta dims: {:?} | input dims: {:?}"
        //          , weights[weight_num].dims().unwrap()
        //          , layers[layer_num].get_delta().dims().unwrap()
        //          , layers[layer_num].get_input().data.dims().unwrap());
        let w_update = af::matmul(&layers[layer_num].get_delta()
                                  , &layers[layer_num].get_input().data.last().unwrap()
                                  , af::MatProp::NONE, af::MatProp::TRANS).unwrap();
        self.velocity_W[velocity_index[0]] = af::mul(&self.momemtum, &self.velocity_W[velocity_index[0]], false).unwrap();
        self.velocity_W[velocity_index[0]] = af::sub(&self.velocity_W[velocity_index[0]]
                                                     , &af::mul(&alpha, &w_update, false).unwrap(), false).unwrap();
        layers[layer_num].set_weights(&af::add(&weights[weight_num]
                                               , &self.velocity_W[velocity_index[0]], false).unwrap()
                                      , weight_num);
        velocity_index[0] += 1;
      }

      // v = momemtum * v + learning_rate * d_b
      // b = b - v
      let biases = layers[layer_num].get_bias();
      for bias_num in (0..biases.len()) {
        let b_update = layers[layer_num].get_delta();
        self.velocity_b[velocity_index[1]] = af::mul(&self.momemtum, &self.velocity_b[velocity_index[1]], false).unwrap();
        self.velocity_b[velocity_index[1]] = af::sub(&self.velocity_b[velocity_index[1]]
                                                     , &af::mul(&alpha, &b_update, false).unwrap(), true).unwrap();
        layers[layer_num].set_bias(&af::sub(&biases[bias_num]
                                            , &self.velocity_b[velocity_index[1]], true).unwrap()
                                   , bias_num);
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
