use af;
use af::{Array, Dim4, MatProp};
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
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array, state: Option<Array>) -> (Array, Option<Array>)
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;

    // z_t = xW [bias added in h_t]
    let wx = af::matmul(&inputs            //activated_input
                        , &ltex.weights[0]
                        , MatProp::NONE
                        , MatProp::NONE);

    // uh_tm1 = htm1 U [bias added in h_t]
    // handle case where time = 0
    let uhtm1 = match ltex.recurrences.len(){
      0 => {
        let output_size = ltex.weights[1].dims()[0]; // is [M x M]
        let init_h_dims = Dim4::new(&[inputs.dims()[0], output_size, 1, 1]);
        if let Some(init_state) = state {
          ltex.recurrences.push(init_state);
        }else {
          ltex.recurrences.push(utils::constant(init_h_dims, inputs.get_type(), 0f32));
        }
        af::matmul(&ltex.recurrences.last().unwrap()
                   , &ltex.weights[1]
                   , MatProp::NONE
                   , MatProp::NONE)
      },
      _ => {
        af::matmul(&ltex.recurrences[ltex.current_unroll]
                   , &ltex.weights[1]
                   , MatProp::NONE
                   , MatProp::NONE)
      }
    };

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
      ltex.recurrences[current_unroll] = h_t.clone();
    }else{                                  // add new
      ltex.inputs.push(inputs.clone());
      ltex.outputs.push(a_t.clone());
      ltex.recurrences.push(h_t.clone());
    }

    // update location in vector
    ltex.current_unroll += 1;

    (a_t.clone(), Some(h_t.clone())) // clone just increases the ref count
  }

  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;
    assert!(current_unroll > 0
            , "Cannot call backward pass without at least 1 forward pass");

    // dz          = grad(a_t)
    // delta_t     = (delta_{t+1} + dh{t+1}) .* dz
    // dh          = delta_{t} * U
    // dW          = x_t^T * delta_t
    // dU          = h_t^T * delta_t
    // db          = sum_{batch} (delta_t)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    let dz = activations::get_derivative(&ltex.activations[0]
                                         , &ltex.outputs[current_unroll - 1]).unwrap();

    // check to see if we already have a state derivative, else add one
    if ltex.state_derivatives.len() == 0 {
      let h_size = ltex.recurrences[current_unroll - 1].dims();
      let h_type = ltex.recurrences[current_unroll - 1].get_type();
      ltex.state_derivatives.push(utils::constant(h_size, h_type, 0.0f32));
    }

    // update the delta for the current time-step by combining with state derivative
    let delta_t = af::mul(&af::add(&ltex.state_derivatives[0]
                                   , delta, false), &dz, false);

    //println!("dh dims = {:?} | delta dims = {:?} | delta prop dims = {:?}", dh.dims(), delta_t.dims(), delta.dims());
    let dw = af::matmul(&ltex.inputs[current_unroll - 1], &delta_t       // delta_w = delta_t * a_{t}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let du = af::matmul(&ltex.recurrences[current_unroll - 1], &delta_t  // delta_u = delta_t * h_{t-1}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let db = af::transpose(&af::sum(&delta_t, 0), false);                // delta_b = \sum_{batch} delta
    ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);
    ltex.deltas[1] = af::add(&ltex.deltas[1], &du, false);
    ltex.deltas[2] = af::add(&ltex.deltas[2], &db, false);

    // add the current state derivative in
    ltex.state_derivatives[0] = af::matmul(&delta_t, &ltex.weights[1], af::MatProp::NONE, af::MatProp::TRANS);

    // update location in vector
    ltex.current_unroll -= 1;

    //println!("dims = {:?}", af::matmul(&delta_t, &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS).dims());
    af::matmul(&delta_t, &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS)    // delta_{t-1}
  }
}
