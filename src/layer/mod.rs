pub use self::dense::Dense;
mod dense;

// pub use self::lstm::LSTM;
// mod lstm;

use af::Dim4;
use af::Array;

#[derive(Clone)]
pub struct Input {
  pub data: Array,
  pub activation: &'static str,
}

pub trait Layer {
  fn forward(&mut self, input: &Input) -> Input;
  fn backward(&mut self, delta: &Array) -> Array;
  fn get_delta(&self) -> Array;
  fn get_weights(&self) -> Vec<Array>;
  fn set_weights(&mut self, weight: &Array, index: usize);
  fn get_bias(&self) -> Vec<Array>;
  fn set_bias(&mut self, bias: &Array, index: usize);
  fn get_bias_dims(&self) -> Vec<Dim4>;
  fn get_weight_dims(&self) -> Vec<Dim4>;
  fn get_input(&self) -> Input;
  fn get_activation_type(&self) -> &'static str;
  fn input_size(&self) -> u64;
  fn output_size(&self) -> u64;
}

  // fn new(input_size: u64, output_size: u64,
  //        output_activation: &'static str, w_init: &'static str, b_init: &str) -> Self where Self: Sized;

pub trait RecurrentLayer : Layer {
  fn new(input_size: u64, output_size: u64
         , inner_activation: &str, outer_activation: &str
         , w_init: &str, b_init: &str) -> Self where Self: Sized;
}
