pub use self::dense::Dense;
mod dense;

// pub use self::lstm::LSTM;
// mod lstm;

use af::Dim4;
use af::Array;

pub trait Layer {
  fn new(input_size: u64, output_size: u64,
         output_activation: &str, w_init: &str, b_init: &str) -> Self where Self: Sized;
  fn forward(&self, activation: &Array) -> Array;
  fn backward(&mut self, inputs: &Array, gradients: &Array, train: bool) -> &Array;
  fn get_weights(&self) -> &Vec<Array>;
  fn get_bias(&self) -> &Vec<Array>;
  fn get_bias_dims(&self) -> &Vec<Dim4>;
  fn get_weight_dims(&self) -> &Vec<Dim4>;
}

pub trait RecurrentLayer {
  fn new(input_size: u64, output_size: u64
         , inner_activation: &str, outer_activation: &str
         , w_init: &str, b_init: &str) -> Self where Self: Sized;
  fn forward(&self, activation: &Array) -> Array;
  fn backward(&mut self, inputs: &Array, gradients: &Array, train: bool) -> &Array;
  fn get_weights(&self) -> &Vec<Array>;
  fn get_bias(&self) -> &Vec<Array>;
  fn get_bias_dims(&self) -> &Vec<Dim4>;
  fn get_weight_dims(&self) -> &Vec<Dim4>;
}

#[derive(Clone)]
pub struct Weights {
  weights : Vec<Array>,
  weight_dims: Vec<Dim4>,
}

#[derive(Clone)]
pub struct Bias {
  bias : Vec<Array>,
  bias_dims: Vec<Dim4>,
}
