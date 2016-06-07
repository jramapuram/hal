use af;
use af::{Array, MatProp};
use std::sync::{Arc, Mutex};

use activations;
use params::{Input, Params};
use layer::Layer;

pub struct Dense {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for Dense
{
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Input, train: bool)-> Input
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();

    // z_t = xW + b [the bias is added in parallel for batch]
    let wx = af::matmul(&inputs.data            // activated_input
                        , &ltex.weights[0]      // layer weights
                        , MatProp::NONE
                        , MatProp::NONE);

    let z_t = af::transpose(&af::add(&af::transpose(&wx, false)
                                     , &ltex.biases[0], true), false);

    // a_t = sigma(z_t)
    let a_t = Input{ data: activations::get_activation(&ltex.activations[0], &z_t).unwrap()
                     , activation: ltex.activations[0].clone() };

    // parameter manager keeps the output & inputs
    // these are only needed for training, so dont store otherwise
    if train {
      ltex.inputs = vec![inputs.clone()];
      ltex.outputs = vec![a_t.clone()];
    }

    a_t.clone() // clone just increases the ref count
  }

  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();

    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    let dz = activations::get_derivative(&ltex.activations[0]
                                         , &ltex.outputs[0].data).unwrap();
    let delta_t = af::mul(delta, &dz, false);
    let dw = af::matmul(&ltex.inputs[0].data, &delta_t // delta_w = delta_t * a_{t-1}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let db = af::transpose(&af::sum(&delta_t, 0), false); // delta_b = delta
    ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);
    ltex.deltas[1] = af::add(&ltex.deltas[1], &db, false);

    af::matmul(&delta_t, &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS)
  }
}
