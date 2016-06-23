use af;
use af::{Array, Dim4, MatProp, identity};
use std::sync::{Arc, Mutex};

use utils;
use activations;
use params::Params;
use layer::Layer;

pub struct RNN {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for RNN
{
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array) -> Array
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;

    // z_t = xW [bias added in h_t]
    let wx = af::matmul(&inputs            //activated_input
                        , &ltex.weights[0]
                        , MatProp::NONE
                        , MatProp::NONE);
    if current_unroll == 0 {
        let output_size = ltex.weights[1].dims()[0];
        let init_h_dims = Dim4::new(&[inputs.dims()[0], output_size, 1, 1]);
        if ltex.recurrences.len() == 0 {
            ltex.recurrences.push(utils::constant(init_h_dims, inputs.get_type(), 0f32));
        }
        else {
            ltex.recurrences[0] = utils::constant(init_h_dims, inputs.get_type(), 0f32);
        }
    }

    let uhtm1 = af::matmul(&ltex.recurrences[current_unroll]
                   , &ltex.weights[1]
                   , MatProp::NONE
                   , MatProp::NONE);

    // h_t = uh_tm1 + wx + b
    let h_t = af::transpose(&af::add(&af::transpose(&uhtm1, false),
                                     &af::add(&af::transpose(&wx, false)
                                              , &ltex.biases[0], true), false)
                            , false);

    // a_t = sigma(h_t)
    let a_t = activations::get_activation(&ltex.activations[0], &h_t).unwrap();

    // parameter manager keeps the output & inputs
    if ltex.inputs.len() > current_unroll { // store in existing
      ltex.inputs[current_unroll] = inputs.clone();
      ltex.outputs[current_unroll] = a_t.clone();
      ltex.recurrences[current_unroll+1] = h_t.clone();
    }else{                                  // add new
      ltex.inputs.push(inputs.clone());
      ltex.outputs.push(a_t.clone());
      ltex.recurrences.push(h_t.clone());
    }

    // update location in vector
    ltex.current_unroll += 1;

    a_t.clone() // clone just increases the ref count
  }

  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array {

    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();

    // update location in vector
    ltex.current_unroll -= 1;

    let current_unroll = ltex.current_unroll;
    assert!(current_unroll >= 0
            , "Cannot call backward pass without at least 1 forward pass");

    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    let dz = activations::get_derivative(&ltex.activations[0]
                                         , &ltex.outputs[current_unroll]).unwrap();
    let delta_t = af::mul(delta, &dz, false);
    
    let maxTime = ltex.outputs.len() - 1;

    // Computation of dL/dh_{t} 
        if current_unroll == ltex.outputs.len()-1 {
            if ltex.deltaRec.len() == 0 {
                ltex.deltaRec.push(delta_t.clone());
            }
            else {
                ltex.deltaRec[0] = delta_t.clone();
            }
        }
        else {
            ltex.deltaRec[0] = af::add(&delta_t, &af::matmul(&ltex.deltaRec[0]
                        , &ltex.weights[1]
                        , af::MatProp::NONE
                        , af::MatProp::NONE), false);
        }

    let dw = af::matmul(&ltex.inputs[current_unroll], &ltex.deltaRec[0]
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let du = af::matmul(&ltex.recurrences[current_unroll], &ltex.deltaRec[0]
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let db = af::transpose(&af::sum(&ltex.deltaRec[0], 0), false);
    ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);
    ltex.deltas[1] = af::add(&ltex.deltas[1], &du, false);
    ltex.deltas[2] = af::add(&ltex.deltas[2], &db, false);

    af::matmul(&ltex.deltaRec[0], &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS)
  }
}
