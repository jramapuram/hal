use af;
use af::{Dim4, Array, MatProp};
use activations::{get_activation_derivative, get_activation};
use initializations::{get_initialization, zeros};
use layer::{ArrayVector, Layer};

pub struct Dense {
  weights: ArrayVector,
  bias: ArrayVector,
  activation: &'static str,
  inputs: Vec<Array>,
}

impl Layer for Dense {
  fn new(input_size: u64, output_size: u64, output_activation: &str, w_init: &str, b_init: &str) -> Dense {
    Dense {
      weights : ArrayVector {
        dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        weights : vec![get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bias: ArrayVector {
        dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        bias : vec![get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1]))],
      },
      activation: output_activation,
      inputs: Vec::new(),
    }
  }

  fn forward(&mut self, activation: &Array) -> Array {
    self.inputs.append(activation);
    self.get_activation(self.activation, &af::matmul(&self.weights[0]
                                                     , activation
                                                     , MatProp::NONE
                                                     , MatProp::NONE).unwrap()
                        + &self.bias[0]);
  }

  fn backward(&mut self, upper_diffs: &Array, gradients: &Array) -> &Array {
    // d_l = (transpose(W) * d_{l+1}) .* dActivation(z) where z = activation w/out non-linearity
    af::mul(&af::matmul(&self.weights[0]
                        , upper_diffs
                        , MatProp::CTRANS
                        , MatProp::NONE).unwrap(), gradients);
  }

  fn get_weights(&self) -> &Vec<Array> {
    &self.weights
  }

  fn set_weights(&mut self, weights: &ArrayVector) {
    self.weights = weights;
  }

  fn get_bias(&self) -> &Vec<Array> {
    &self.bias
  }

  fn set_bias(&mut self, bias: &ArrayVector) {
    self.bias = bias;
  }

  fn get_bias_dims(&self) -> &Vec<Dim4> {
    &self.bias.dims
  }

  fn get_weight_dims(&self) -> &Vec<Dim4> {
    &self.weights.dims
  }

  fn get_inputs(&self) -> &Array {
    &self.inputs
  }
}
