pub use self::dense::Dense;
mod dense;
pub use self::unitary::Unitary;
mod unitary;

pub use self::rnn::RNN;
mod rnn;

// pub use self::lstm::LSTM;
// mod lstm;

use af::Array;
use params::Params;
use std::sync::{Arc, Mutex};

pub trait Layer {
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array) -> Array;
  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array;
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
              , inputs: &Array);     // x_t & h_{t-1}
}
