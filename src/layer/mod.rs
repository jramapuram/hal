mod dense;
//mod lstm;

use af::Dim4;
use af::Array;

pub trait Layer {
  fn new(input_size: u64, output_size: u64) -> Self;
  fn forward(&self, activation: &Array) -> Array;
  //fn backward(&self, inputs: &Array, gradients: &Array);
  //fn connect<'a, 'b>(&'a self, next: &'b Layer);
  fn get_weights(&self) -> &Vec<Array>;
  fn get_bias(&self) -> &Vec<Array>;
  fn get_bias_dims(&self) -> &Vec<Dim4>;
  fn get_weight_dims(&self) -> &Vec<Dim4>;
}

pub struct Params {
  weights : Vec<Array>,
  bias : Vec<Array>,
  weight_dims: Vec<Dim4>,
  bias_dims: Vec<Dim4>,
}
