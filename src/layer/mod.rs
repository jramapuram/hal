pub use self::dense::Dense;
mod dense;
pub use self::unitary::Unitary;
mod unitary;

pub use self::rnn::RNN;
mod rnn;

// pub use self::lstm::LSTM;
// mod lstm;

use af;
use af::{Array, MatProp};
use params::Params;
use std::sync::{Arc, Mutex};

use activations;

pub trait Layer {
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array, state: Option<&Vec<Array>>) -> (Array, Option<Vec<Array>>);
  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array;
}

pub trait RecurrentLayer {
  fn state_size(self) -> usize;
}

pub trait RTRL{
  // fn rtrl(&self, dW_tm1: &mut Array  // previous W derivatives for [I, F, Ct]
  //         , dU_tm1: &mut Array   // previous U derivatives for [I, F, Ct]
  //         , db_tm1: &mut Array   // previous b derivatives for [I, F, Ct]
  //         , z_t: &Array          // current time activation
  //         , inputs: &Array);     // x_t & h_{t-1}
  fn rtrl(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array;
}

/// Helper to run f(Wx + b) where bias is optional
pub fn linear(input: &Array, weight: &Array, bias: Option<&Array>, activation: &str) -> Array
{
  // w_x = xW
  // z_t = w_x + b [the bias is added in parallel for batch]
  let wx = af::matmul(input     // activated_input
                      , weight   // layer weights
                      , MatProp::NONE
                      , MatProp::NONE);
  let z_t = match bias {
    Some(b) => af::transpose(&af::add(&af::transpose(&wx, false)
                                      , b, true), false),
    None    => wx.clone()
  };

  // activation(z_t)
  activations::get_activation(activation, &z_t).unwrap()
}


/// Helper that computes the backward operation on f(Wx + b) and returns delta, dW, db
pub fn linear_backward(delta: &Array, input: &Array
                       , output: &Array, activation: &str) -> (Array, Array, Array)
{
  // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
  // delta_{t-1} = (transpose(W_{t}) * d_{l})
  let dz = activations::get_derivative(activation, output).unwrap();
  let delta_t = af::mul(delta, &dz, false);
  let dw = af::matmul(input
                      , &delta_t                        // delta_w = delta_t * a_{t}
                      , af::MatProp::TRANS
                      , af::MatProp::NONE);
  let db = af::transpose(&af::sum(&delta_t, 0), false); // delta_b = sum_{batch}delta
  return (delta_t.clone(), dw.clone(), db.clone())
}
