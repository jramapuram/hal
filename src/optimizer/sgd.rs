use af;
use af::{Array};
use std::collections::HashMap;
use std::default::Default;

use loss;
use initializations;
use activations;
use layer::{Layer};
use optimizer::{Optimizer, update_delta, grads};

pub struct SGD {
  name: &'static str,
  learning_rate: f32,
  momemtum: f32,
  decay: f32,
  nesterov: bool,
  clip_grad: u64,
  iter: u64,
  velocity_W: Vec<Array>,
  velocity_b: Vec<Array>,
}

impl Default for SGD {
  fn default() -> SGD { //docs say Self
    SGD {
      name: "SGD",
      learning_rate: 0.001,
      momemtum: 0.5,
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

  fn update_parameters(&mut self, layers: &mut Vec<Box<Layer>>, batch_size: u64)
  {
    let alpha = self.learning_rate / batch_size as f32;
    let mut velocity_index = [0, 0];
    for layer_num in 0..layers.len() {
      let (delta_w, delta_b) = layers[layer_num].get_delta();

      // v = momemtum * v + learning_rate * d_w
      // W = W - v
      let weights = layers[layer_num].get_weights();
      for weight_num in (0..weights.len()) {
        self.velocity_W[velocity_index[0]] = af::mul(&self.momemtum, &self.velocity_W[velocity_index[0]]).unwrap();
        self.velocity_W[velocity_index[0]] = af::sub(&self.velocity_W[velocity_index[0]]
                                                  , &af::mul(&alpha, &delta_w).unwrap()).unwrap();
        layers[layer_num].set_weights(&af::add(&weights[weight_num]
                                               , &self.velocity_W[velocity_index[0]]).unwrap()
                                      , weight_num);
        velocity_index[0] += 1;
      }

      // v = momemtum * v + learning_rate * d_b
      // b = b - v
      let biases = layers[layer_num].get_bias();
      for bias_num in (0..biases.len()) {
        self.velocity_b[velocity_index[1]] = af::mul(&self.momemtum, &self.velocity_b[velocity_index[1]]).unwrap();
        self.velocity_b[velocity_index[1]] = af::sub(&self.velocity_b[velocity_index[1]]
                                                    , &af::mul(&alpha, &delta_b).unwrap()).unwrap();
        layers[layer_num].set_bias(&af::sub(&biases[bias_num]
                                            , &self.velocity_b[velocity_index[1]]).unwrap()
                                   , bias_num);
        velocity_index[1] += 1;
      }
    }
  }


  fn optimize(&mut self, layers: &mut Vec<Box<Layer>>
              , prediction: &Array
              , target: &Array
              , loss: &'static str) -> f32
  {
    self.iter += 1;
    self.learning_rate *= 1.0 / (1.0 + self.decay * (self.iter as f32));
    let last_index = layers.len() - 1;
    let prev_activation = layers[last_index].get_input();
    let diffs = grads(prediction
                      , target
                      , loss
                      , layers[last_index].get_activation_type());
    update_delta(&mut layers[last_index], &prev_activation, &diffs);

    for i in (1..last_index).rev() {
      // d_l = (W_{l+1}^T * d_{l+1}) .* derivative(z) where z = activation w/out non-linearity
      let prev_activation = layers[i].get_input();
      let grad = activations::get_activation_derivative(layers[i].get_activation_type(), &prev_activation).unwrap();
      let diffs = layers[i].backward(&diffs, &grad);
      update_delta(&mut layers[i], &prev_activation, &diffs);
    }

    loss::get_loss(loss, prediction, target).unwrap()
  }

  fn info(&self){
    println!("name:          {}", self.name);
    println!("learning_rate: {}", self.name);
    println!("momemtum:      {}", self.name);
    println!("decay:         {}", self.name);
    println!("nesterov:      {}", self.name);
    println!("clip_grad:     {}", self.name);
  }
}
