pub use self::dense::Dense;
mod dense;

// pub use self::lstm::LSTM;
// mod lstm;

use af::Dim4;
use af::Array;

#[derive(Clone)]
pub struct ArrayVector {
  data: Vec<Array>,
}

pub trait Layer {
  fn new(input_size: u64, output_size: u64,
         output_activation: &'static str, w_init: &'static str, b_init: &str) -> Self where Self: Sized;
  fn forward(&mut self, activation: &Array) -> Array;
  fn backward(&self, inputs: &Array, gradients: &Array) -> Array;
  fn get_weights(&self) -> &Vec<Array>;
  fn set_weights(&mut self, weight: Array, index: usize);
  fn get_bias(&self) -> &Vec<Array>;
  fn set_bias(&mut self, bias: Array, index: usize);
  fn get_bias_dims(&self) -> Vec<Dim4>;
  fn get_weight_dims(&self) -> Vec<Dim4>;
  fn get_inputs(&self) -> &(Array, Array);
  fn get_activation_type(&self) -> &'static str;
}

pub trait RecurrentLayer : Layer {
  fn new(input_size: u64, output_size: u64
         , inner_activation: &str, outer_activation: &str
         , w_init: &str, b_init: &str) -> Self where Self: Sized;
}
