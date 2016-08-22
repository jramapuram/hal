use af;
use af::{Array};
use std::sync::{Arc, Mutex};

use layer;
use layer::{Layer};
use params::Params;

pub struct Dense {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for Dense
{
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array, state: Option<&Vec<Array>>) -> (Array, Option<Vec<Array>>)
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();

    // output is simple linear operation (followed by added non-linearity)
    let a_t = layer::linear(inputs, &ltex.weights[0], Some(&ltex.biases[0]), &ltex.activations[0]);

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

  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;
    assert!(current_unroll > 0
            , "Cannot call backward pass without at least 1 forward pass");

    // utilize the helper to get our deltas
    let (delta_t, dw, db) = layer::linear_backward(delta
                                                   , &ltex.inputs[current_unroll - 1]
                                                   , &ltex.outputs[current_unroll - 1]
                                                   , &ltex.activations[0]);
    ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);
    ltex.deltas[1] = af::add(&ltex.deltas[1], &db, false);

    ltex.current_unroll -= 1;

    af::matmul(&delta_t, &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS)
  }
}
