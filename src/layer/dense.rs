use af;
use af::{Dim4, Array, MatProp};

use activations;
use initializations;
use layer::{Layer, Input};

#[allow(non_snake_case)]
pub struct Dense {
  weights: Vec<Array>,
  bias: Vec<Array>,
  delta: Array,
  inputs: Input,
  activation: &'static str,
}

impl Dense {
  pub fn new(input_size: u64, output_size: u64
         , output_activation: &'static str
         , w_init: &'static str, b_init: &str) -> Dense
  {
    Dense {
      weights: vec![initializations::get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1])).unwrap()],                                // W
      bias:    vec![initializations::get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()],                                         // b
      inputs:  Input{data: initializations::get_initialization("zeros", Dim4::new(&[input_size, 1, 1, 1])).unwrap(), activation: output_activation},   // z_{l-1} | activ{z_{l-1}}
      delta: initializations::get_initialization("zeros", Dim4::new(&[output_size, 1, 1, 1])).unwrap(),                                                // delta
      activation: output_activation,
    }
  }
}

impl Layer for Dense {

  fn forward(&mut self, input: &Input) -> Input {
    // keep previous_activation
    self.inputs = input.clone();

    // apply the activation to the previous layer [Optimization: Memory saving]
    let activated_input = activations::get_activation(input.activation, &input.data).unwrap();

    // sigma(Wx + b)
    let mul = af::matmul(&self.weights[0]
                         , &activated_input
                         , MatProp::NONE
                         , MatProp::NONE).unwrap();
    Input {data: af::add(&mul, &self.bias[0], true).unwrap(), activation: self.activation}
  }

  fn backward(&mut self, delta: &Array) -> Array {
    // d_l = (transpose(W) * d_{l}) .* dActivation(z-1) where z = activation w/out non-linearity
    self.delta = delta.clone();
    let activation_prev = activations::get_activation(self.inputs.activation, &self.inputs.data).unwrap();
    let d_activation_prev = activations::get_activation_derivative(self.inputs.activation, &activation_prev).unwrap();
    let delta_prev = af::mul(&af::matmul(&self.weights[0], delta, af::MatProp::TRANS, af::MatProp::NONE).unwrap()
                             , &d_activation_prev, false).unwrap();
    delta_prev
  }

  fn get_delta(&self) -> Array {
    self.delta.clone()
  }

  fn get_weights(&self) -> Vec<Array> {
    self.weights.clone()
  }

  fn set_weights(&mut self, weights: &Array, index: usize) {
    self.weights[index] = weights.clone();
  }

  fn get_bias(&self) -> Vec<Array> {
    self.bias.clone()
  }

  fn set_bias(&mut self, bias: &Array, index: usize) {
    self.bias[index] = bias.clone();
  }

  fn get_bias_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for b in &self.bias {
      dims.push(b.dims().unwrap().clone())
    }
    dims
  }

  fn get_weight_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for w in &self.weights {
      dims.push(w.dims().unwrap().clone())
    }
    dims
  }

  fn get_input(&self) -> Input {
    self.inputs.clone()
  }

  fn output_size(&self) -> u64 {
    let weight_dims = self.get_weight_dims();
    weight_dims[weight_dims.len() - 1][1]    
  }

  fn input_size(&self) -> u64 {
    let weight_dims = self.get_weight_dims();
    weight_dims[0][0]
  }

  fn get_activation_type(&self) -> &'static str {
    &self.activation
  }
}
