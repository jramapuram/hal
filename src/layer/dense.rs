use af;
use af::{Array, MatProp};
use std::sync::{Arc, Mutex};

use activations;
use params::Params;
use layer::Layer;

pub struct Dense {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for Dense
{
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array, state: Option<Array>) -> (Array, Option<Array>)
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();

    // z_t = xW + b [the bias is added in parallel for batch]
    let wx = af::matmul(&inputs              // activated_input
                        , &ltex.weights[0]   // layer weights
                        , MatProp::NONE
                        , MatProp::NONE);

    let z_t = af::transpose(&af::add(&af::transpose(&wx, false)
                                     , &ltex.biases[0], true), false);

    // a_t = sigma(z_t)
    let a_t = activations::get_activation(&ltex.activations[0], &z_t).unwrap();

    // parameter manager keeps the output & inputs
    let current_unroll = ltex.current_unroll;
    if ltex.inputs.len() > current_unroll { // store in existing
      ltex.inputs[current_unroll] = inputs.clone();
      ltex.outputs[current_unroll] = a_t.clone();
    }else{                                  // add new
      ltex.inputs.push(inputs.clone());
      ltex.outputs.push(a_t.clone());
    }

    // update location in vector
    ltex.current_unroll += 1;

    (a_t.clone(), None) // clone just increases the ref count
  }

  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array, state_delta: Option<Array>) -> (Array, Option<Array>)
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;
    assert!(current_unroll > 0
            , "Cannot call backward pass without at least 1 forward pass");

    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    let dz = activations::get_derivative(&ltex.activations[0]
                                         , &ltex.outputs[current_unroll - 1]).unwrap();
    let delta_t = af::mul(delta, &dz, false);
    let dw = af::matmul(&ltex.inputs[current_unroll - 1]
                        , &delta_t                        // delta_w = delta_t * a_{t}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let db = af::transpose(&af::sum(&delta_t, 0), false); // delta_b = sum_{batch}delta

    ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);
    ltex.deltas[1] = af::add(&ltex.deltas[1], &db, false);

    ltex.current_unroll -= 1;

    (af::matmul(&delta_t, &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS), None)
  }
}
