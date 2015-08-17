use af;
use af::{Dim4, Array};
use af::MatProp;
use activations;
use initializations;
use layer::{Weights, Bias, Layer};

pub struct Dense {
  weights: Weights,
  bias: Bias,
  diffs: Array,
  activation: &str,
}

impl Layer for Dense {
  fn new(input_size: u64, output_size: u64, output_activation: &str, w_init: &str, b_init: &str) -> Dense {
    Dense {
      weights : Weights {
        dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        weights : vec![get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bias: Bias {
        dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        bias : vec![get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1]))],
      }
      diffs: zeros(Dim4::new(&[output_size, 1, 1, 1])),
      activation: output_activation
    }
  }

  fn forward(&self, activation: &Array) -> Array {
    get_activation(self.activation, &af::matmul(&self.weights[0], activation, MatProp::NONE, MatProp::NONE).unwrap() + &self.bias[0])
  }

  fn backward(&mut self, upper_diffs: &Array, gradients: &Array, train: bool) -> &Array {
    if train{
      self.diffs = &af::matmul(&self.weights[0], upper_diffs, MatProp::NONE, MatProp::NONE).unwrap() * gradients;
    }
   &self.diffs
  }

  fn get_weights(&self) -> &Vec<Array> {
    &self.weights
  }

  fn get_bias(&self) -> &Vec<Array> {
    &self.bias
  }

  fn get_bias_dims(&self) -> &Vec<Dim4> {
    &self.p.bias_dims
  }

  fn get_weight_dims(&self) -> &Vec<Dim4> {
    &self.params.weight_dims
  }
}
