use af;
use activations;
use initializations;
use af::{Dim4, Array, MatProp};
use layer::{ArrayVector, Layer};

pub struct Dense {
  weights: ArrayVector,
  bias: ArrayVector,
  activation: &'static str,
  inputs: Vec<(Array, Array)>,
}

impl Layer for Dense {
  fn new(input_size: u64, output_size: u64, output_activation: &'static str, w_init: &'static str, b_init: &str) -> Dense {
    Dense {
      weights : ArrayVector {
        data : vec![initializations::get_initialization(w_init, &Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bias: ArrayVector {
        data : vec![initializations::get_initialization(b_init, &Dim4::new(&[output_size, 1, 1, 1]))],
      },
      activation: output_activation,
      inputs: Vec::new(),
    }
  }

  fn forward(&mut self, activation: &Array) -> Array {
    // append tuple: (previous_input, Wx + b)
    self.inputs.push((*activation, af::add(af::matmul(&self.weights.data[0]
                                                      , activation
                                                      , MatProp::NONE
                                                      , MatProp::NONE).unwrap()
                                          , self.bias.data[0]).unwrap()));
    //sigma(Wx + b)
    activations::get_activation(self.activation, &self.inputs.last().unwrap().1)    
  }

  fn backward(&self, upper_diffs: &Array, gradients: &Array) -> Array {
    // d_l = (transpose(W) * d_{l+1}) .* dActivation(z) where z = activation w/out non-linearity
    af::mul(af::matmul(&self.weights.data[0]
                       , upper_diffs
                       , MatProp::CTRANS
                       , MatProp::NONE).unwrap()
            , *gradients).unwrap()
  }

  fn reset(&mut self) {
    self.inputs.clear();
  }

  fn get_weights(&self) -> &Vec<Array> {
    &self.weights.data
  }

  fn set_weights(&mut self, weights: ArrayVector) {
    self.weights = weights;
  }

  fn get_bias(&self) -> &Vec<Array> {
    &self.bias.data
  }

  fn set_bias(&mut self, bias: ArrayVector) {
    self.bias = bias;
  }

  fn get_bias_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for b in self.bias.data {
      dims.push(b.dims().unwrap())
    }
    dims
  }

  fn get_weight_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for w in self.weights.data {
      dims.push(w.dims().unwrap())
    }
    dims
  }

  fn get_inputs(&self) -> &Vec<(Array, Array)> {
    &self.inputs
  }

  fn get_activation_type(&self) -> &'static str {
    &self.activation
  }
}
