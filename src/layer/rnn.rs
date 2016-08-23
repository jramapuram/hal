use af;
use af::{Array, Dim4, MatProp};
use std::sync::{Arc, Mutex};

use utils;
use layer;
use activations;
use params::{Params, RNNIndex};
use layer::{Layer, RecurrentLayer};

pub struct RNN {
  pub input_size: usize,
  pub hidden_size: usize,
  pub output_size: usize,
}

impl RecurrentLayer for RNN {
  fn state_size(self) -> usize {
    self.output_size
  }
}

impl RNN
{
  /// A helper to do a large matmul if possible
  fn optimized_state_calc(&self, input: &Array, a_tm1: &Array
                          , weight_i2h: &Array, weight_h2h: &Array
                          , hidden_bias: &Array, hidden_activation: &String) -> Array
  {
    let idims = weight_i2h.dims();
    let hdims = weight_h2h.dims();
    // There are two cases:
    //   1. idims == hdims    [most optimized]
    //   2. idims != hdims    [least optimized]
    match idims == hdims {
      true => {
        let is_vec = vec![input, a_tm1];
        let wu_vec = vec![weight_i2h, weight_h2h];
        layer::linear(&af::join_many(1, is_vec)
                      , &af::join_many(0, wu_vec)
                      , Some(hidden_bias)
                      , hidden_activation)
      },

      false => {
        // calculate the input to hidden mapping
        let wx = af::matmul(input            //activated_input
                            , weight_i2h
                            , MatProp::NONE
                            , MatProp::NONE);

        // add the i2h map to the bias (as they are the same size)
        // this helps use the linear projection operator
        let wx_p_b = af::add(&af::transpose(&wx, false)
                             , hidden_bias, true);
        layer::linear(a_tm1, weight_h2h, Some(&wx_p_b), hidden_activation)
      }
    }
  }
}


impl Layer for RNN
{
  fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array, state: Option<&Vec<Array>>) -> (Array, Option<Vec<Array>>)
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;

    // set a_{t-1} to the given state (if provided)
    // if state is provided ensure to save it as the last recurrence
    //   --> Note: this will override the last recurrence if it exists
    // also handle case where time = 0 for both cases
    let atm1 = match state {
      Some(init_state)  => match ltex.recurrences.len() {
        0 => {
          ltex.recurrences.push(init_state[0].clone());
          init_state[0].clone() // only one recurrence for vanilla RNN
        },

        _ => {
          let rlen = ltex.recurrences.len();
          ltex.recurrences[rlen - 1] = init_state[0].clone();
          init_state[0].clone() // only one recurrence for vanilla RNN
        },
      },

      None              => match ltex.recurrences.len() {
        0 => {
          let output_size = ltex.weights[RNNIndex::HiddenToHidden as usize].dims()[0]; // is [M x M]
          let init_h_dims = Dim4::new(&[inputs.dims()[0], output_size, 1, 1]);
          let zero_state = utils::constant(init_h_dims, inputs.get_type(), 0f32);
          ltex.recurrences.push(zero_state.clone());
          zero_state
        },
        _ => ltex.recurrences.last().unwrap().clone()
      }
    };

    // compute the current state a_t in an optimized fashion [if possible]
    let a_t = self.optimized_state_calc(inputs, &atm1
                                        , &ltex.weights[RNNIndex::InputToHidden as usize]
                                        , &ltex.weights[RNNIndex::HiddenToHidden as usize]
                                        , &ltex.biases[RNNIndex::InputToHidden as usize]
                                        , &ltex.activations[0]);

    // calculate the output projection
    // v_t = V*a_t + b_v
    // o_t = outer_activation(v_t)
    let o_t = layer::linear(&a_t, &ltex.weights[RNNIndex::HiddenToOutput as usize]
                            , Some(&ltex.biases[RNNIndex::HiddenToOutput as usize])
                            , &ltex.activations[1]);

    // parameter manager keeps the output & inputs
    if ltex.inputs.len() > current_unroll { // store in existing
      ltex.inputs[current_unroll] = inputs.clone();
      ltex.outputs[current_unroll] = o_t.clone();
      ltex.recurrences[current_unroll + 1] = a_t.clone();
    }else{                                  // add new
      ltex.inputs.push(inputs.clone());
      ltex.outputs.push(o_t.clone());
      ltex.recurrences.push(a_t.clone());
    }

    // update location in vector
    ltex.current_unroll += 1;

    (o_t.clone(), Some(vec![a_t.clone()])) // clone just increases the ref count
  }

  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array
  {
    // get a handle to the underlying params
    let mut ltex = params.lock().unwrap();
    let current_unroll = ltex.current_unroll;
    assert!(current_unroll > 0
            , "Cannot call backward pass without at least 1 forward pass");

    // compute the derivatives of the output projection layer
    let (mut delta_v, dv, db_h2o) = layer::linear_backward(delta
                                                       , &ltex.recurrences[current_unroll]
                                                       , &ltex.outputs[current_unroll - 1]
                                                       , &ltex.activations[1]);
    delta_v = af::matmul(&delta_v, &ltex.weights[RNNIndex::HiddenToOutput as usize]
                         , af::MatProp::NONE, af::MatProp::TRANS);

    // check to see if we already have a state derivative, else add one
    if ltex.state_derivatives.len() == 0 {
      let h_size = ltex.recurrences[0].dims();
      let h_type = ltex.recurrences[0].get_type();
      ltex.state_derivatives.push(utils::constant(h_size, h_type, 0.0f32));
    }

    // dz          = grad(a_t)
    // delta_t     = (delta_{t+1} + dh{t+1}) .* dz
    // dh          = delta_{t} * U
    // dW          = x_t^T * delta_t
    // dU          = h_t^T * delta_t
    // db          = sum_{batch} (delta_t)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    let dz = activations::get_derivative(&ltex.activations[0]
                                         , &ltex.recurrences[current_unroll]).unwrap();

    // update the delta for the current time-step by combining with state derivative
    let delta_t = af::mul(&af::add(&ltex.state_derivatives[0]
                                   , &delta_v, false), &dz, false);

    let dw = af::matmul(&ltex.inputs[current_unroll - 1], &delta_t       // delta_w = delta_t * a_{t}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let du = af::matmul(&ltex.recurrences[current_unroll - 1], &delta_t  // delta_u = delta_t * h_{t-1}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE);
    let db_i2h = af::transpose(&af::sum(&delta_t, 0), false);            // delta_b = \sum_{batch} delta

    // push in the appropriate gradients
    ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);     // i2h
    ltex.deltas[1] = af::add(&ltex.deltas[1], &dv, false);     // h2o
    ltex.deltas[2] = af::add(&ltex.deltas[2], &du, false);     // h2h
    ltex.deltas[3] = af::add(&ltex.deltas[3], &db_i2h, false); // i2h bias
    ltex.deltas[4] = af::add(&ltex.deltas[4], &db_h2o, false); // h2o bias

    // add the current state derivative in
    ltex.state_derivatives[0] = af::matmul(&delta_t
                                           , &ltex.weights[RNNIndex::HiddenToHidden as usize]
                                           , af::MatProp::NONE, af::MatProp::TRANS);

    // update location in vector
    ltex.current_unroll -= 1;

    // delta_{t-1}
    af::matmul(&delta_t, &ltex.weights[0], af::MatProp::NONE, af::MatProp::TRANS)
  }
}
