pub use self::dense::Dense;
mod dense;

// pub use self::lstm::LSTM;
// mod lstm;

use af::Array;

use params::{Input, Params};

pub trait Layer {
  fn forward(&self, params: &mut Params, input: &Input, recurrence: &Input) -> (Input, Input);
  fn backward(&self, params: &mut Params, delta: &Array) -> Array;
  // fn get_delta(&self) -> Array;
  // fn get_weights(&self) -> Vec<Array>;
  // fn set_weights(&mut self, weight: &Array, index: usize);
  // fn get_bias(&self) -> Vec<Array>;
  // fn set_bias(&mut self, bias: &Array, index: usize);
  // fn get_bias_dims(&self) -> Vec<Dim4>;
  // fn get_weight_dims(&self) -> Vec<Dim4>;
  // fn get_input(&self) -> Input;
  // fn get_activation_type(&self) -> &'static str;
  // fn input_size(&self) -> u64;
  // fn output_size(&self) -> u64;
}

pub trait RecurrentLayer: Layer {
  fn new(input_size: u64, output_size: u64
         , inner_activation: &str, outer_activation: &str
         , w_init: &str, b_init: &str) -> Self where Self: Sized;
}

pub trait RTRL{
  fn rtrl(&self, dW_tm1: &mut Array  // previous W derivatives for [I, F, Ct]
              , dU_tm1: &mut Array   // previous U derivatives for [I, F, Ct]
              , db_tm1: &mut Array   // previous b derivatives for [I, F, Ct]
              , z_t: &Array          // current time activation
              , inputs: &Input);     // x_t & h_{t-1}
}
