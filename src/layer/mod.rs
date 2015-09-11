pub use self::dense::Dense;
mod dense;

// pub use self::lstm::LSTM;
// mod lstm;

use af::Dim4;
use af::Array;

pub trait Layer {
  fn new(input_size: u64, output_size: u64,
         output_activation: &'static str, w_init: &'static str, b_init: &str) -> Self where Self: Sized;
  fn forward(&mut self, activation: &Array) -> Array;
  fn backward(&self, inputs: &Array, gradients: &Array) -> Array;
  fn update(&mut self, delta: (Array, Array), train: bool);
  fn get_delta(&self) -> (Array, Array);
  fn get_weights(&self) -> Vec<Array>;
  fn set_weights(&mut self, weight: &Array, index: usize);
  fn get_bias(&self) -> Vec<Array>;
  fn set_bias(&mut self, bias: &Array, index: usize);
  fn get_bias_dims(&self) -> Vec<Dim4>;
  fn get_weight_dims(&self) -> Vec<Dim4>;
  fn get_input(&self) -> Array;
  fn get_activation_type(&self) -> &'static str;
  fn input_size(&self) -> u64;
  fn output_size(&self) -> u64;
}

pub trait RecurrentLayer : Layer {
  fn new(input_size: u64, output_size: u64
         , inner_activation: &str, outer_activation: &str
         , w_init: &str, b_init: &str) -> Self where Self: Sized;
}
