use af;
use af::{Dim4, Array};
use af::MatProp;
use layer::{Params, Layer};

pub struct Dense {
  params: Params,
}

impl Layer for Dense {
  fn new(input_size: u64, output_size: u64) -> Dense {
    Dense {
      params : Params {
        weight_dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        bias_dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        weights : vec![af::randn(Dim4::new(&[input_size, output_size, 1, 1]), af::Aftype::F32).unwrap()],
        bias : vec![af::constant(1.0 as f32, Dim4::new(&[output_size, 1, 1, 1])).unwrap()],
      },
    }
  }

  fn forward(&self, activation: &Array) -> Array {
    &af::matmul(&self.params.weights[0], activation, MatProp::CTRANS, MatProp::CTRANS).unwrap() + &self.params.bias[0]
  }

  // pub fn backward(&self, inputs: &Array, gradients: &Array) {
     
  //}

  fn get_weights(&self) -> &Vec<Array> {
    &self.params.weights
  }

  fn get_bias(&self) -> &Vec<Array> {
    &self.params.bias
  }

  fn get_bias_dims(&self) -> &Vec<Dim4> {
    &self.params.bias_dims
  }

  fn get_weight_dims(&self) -> &Vec<Dim4> {
    &self.params.weight_dims
  }
}
