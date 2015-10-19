use af;
use af::{Dim4, Array};
use af::MatProp;

use activations;
use initializations;
use params::{LSTMIndex, GatedRecurrence, Input, Params};

pub struct LSTM {
  pub input_size: usize,
  pub output_size: usize,
  pub return_sequences: bool,
}

pub enum ActivationIndex {
  Inner,
  Outer,
}

impl RTRL for LSTM {
  pub fn rtrl(&self, delta: &Array, params: &mut Params) -> Array
  {
    let inner_activation = params.activation[0];
    let outer_activation = params.activation[1];
    let i_t  = params.recurrences[LSTMIndex::Input];
    let f_t  = params.recurrences[LSTMIndex::Forget];
    let o_t  = params.recurrences[LSTMIndex::Output];
    let ct_t = params.recurrences[LSTMIndex::CellTilda];
    let c_t  = params.recurrences[LSTMIndex::Cell];
    let h_t  = params.recurrences[LSTMIndex::CellOutput];

    let inputs = params.inputs.last().unwrap();
    let mut derivatives = params.optional.pop().unwrap();
    let mut dW_tm1 = derivatives[0];
    let mut dU_tm1 = derivatives[1];
    let mut db_tm1 = derivatives[2];

    // e_t = o_t * outer_activation'(c_t) * delta
    let e_t = af::mul(&af::mul(&o_t, &activations::get_activation_derivative(outer_activation, &c_t).unwrap()).unwrap()
                               , delta).unwrap();

    // compute their derivatives [diff(z_i), diff(z_f), diff(z_ct)]
    let dz = vec![&activations::get_activation_derivative(inner_activation, &i_t).unwrap()
                  , &activations::get_activation_derivative(inner_activation, &f_t).unwrap()
                  , &activations::get_activation_derivative(.outer_activation, &ct_t).unwrap()];
    let ct_ctm1_it = vec![&ct_t, &c_tm1, &i_t];

    // [Ct_t; C_{t-1}; i_t] * dz
    let dzprod = af::mul(&af::join_many(0, ct_ctm1_it).unwrap()
                         , af::join_many(0, dzvec).unwrap(), false).unwrap();

    // dC_t/dWi  = (dC_{t-1}/dWi  * f_t) + ct_t  * inner_activation(z_i) * x_t
    // dC_t/dWf  = (dC_{t-1}/dWf  * f_t) + c_tm1 * inner_activation(z_f) * x_t
    // dC_t/dWct = (dC_{t-1}/dWct * f_t) + i_t   * outer_activation(Ct)  * x_t
    let w_lhs = af::mul(dW_tm1, &f_t, true).unwrap(); // dC_{t-1}/dW * f_t
    let w_rhs = af::mul(&dzprod, &inputs.data, true).unwrap();
    dW_tm1 = af::add(&w_lhs, &w_rhs, false).unwrap();

    // dC_t/dUi = (dC_{t-1}/dUi * f_t) + ct_t * inner_activation(z_i) * h_{t-1}
    // dC_t/dUf = (dC_{t-1}/dUf * f_t) + c_tm1 * inner_activation(z_f) * h_{t-1}
    // dC_t/dUct = (dC_{t-1}/dUct * f_t) + outer_activation(Ct) * x_t * h_{t-1}
    let u_lhs = af::mul(dU_tm1, &f_t, true).unwrap(); // dC_{t-1}/dU * f_t
    let u_rhs = af::mul(&dzprod, &recurrences.data, true).unwrap();
    dU_tm1 = af::add(&u_lhs, &u_rhs, false).unwrap();

    // dC_t/dbi = (dC_{t-1}/dbi * f_t) + ct_t * inner_activation(z_i)
    // dC_t/dbf = (dC_{t-1}/dbf * f_t) + c_tm1 * inner_activation(z_f)
    // dC_t/dbct = (dC_{t-1}/dbct * f_t) + outer_activation(Ct)
    let b_lhs = af::mul(db_tm1, &f_t, true).unwrap(); // dC_{t-1}/db * f_t
    params.optional[2] = af::add(&b_lhs, &dzprod, false).unwrap(); //db_{t-1}
  }
}

impl Layer for LSTM {
  fn forward(&self, params: &mut Params, inputs: &Input) -> Input
  {
    assert!(inputs.data.dims().unwrap()[2] == 1); // only planar data here

    // keep previous layer's outputs
    params.inputs.push(inputs.clone());

    let h_tm1 = params.recurrences[LSTMIndex::CellOutput].last().unwrap();  // cell output @ t-1
    let c_tm1 = params.recurrences[LSTMIndex::Cell].last().unwrap();        // cell memory @ t-1
    let x_t   = params.inputs.last().unwrap();                              // x_t
    let inner_activation = params.activations[0];
    let outer_activation = params.activations[1];

    // forward pass in a batch for performance
    let weights_ref    = vec![&params.weights[LSTMIndex::Input]
                              , &params.weights[LSTMIndex::Forget]
                              , &params.weights[LSTMIndex::Output]
                              , &params.weights[LSTMIndex::CellTilda]];
    let offset = 4; // the offset from weights --> recurrent weights
    let recurrents_ref = vec![&params.weights[LSTMIndex::Input as usize + offset]
                              , &params.weights[LSTMIndex::Forget as usize + offset]
                              , &params.weights[LSTMIndex::Output as usize + offset]
                              , &params.weights[LSTMIndex::CellTilda as usize + offset]];
    let bias_ref       = vec![&params.biases[LSTMIndex::Input]
                              , &params.biases[LSTMIndex::Forget]
                              , &params.biases[LSTMIndex::Output]
                              , &params.biases[LSTMIndex::CellTilda]];
    // [z(i,f,o,ct)_t] = W*x + U*h_tm1 + b
    let z_t = af::add(&af::add(&af::matmul(&af::join_many(0, weights_ref).unwrap(), &x_t.data).unwrap()
                               , &af::matmul(&af::join_many(0, recurrents_ref).unwrap(), &h_tm1).unwrap(), false).unwrap()
                      , &af::join_many(0, bias_ref).unwrap(), true).unwrap();
    let i_t   = activations::get_activation(inner_activation, &af::rows(&z_t, 0, 1).unwrap());
    let f_t   = activations::get_activation(inner_activation, &af::rows(&z_t, 1, 2).unwrap());
    let o_t   = activations::get_activation(inner_activation, &af::rows(&z_t, 2, 3).unwrap());
    let ct_t  = activations::get_activation(inner_activation, &af::rows(&z_t, 3, 4).unwrap());

    // C_{t} = i_{t} * Ct_{t} + f_{t} * C_{t-1}
    // h_{t} = o_{t} * outer_activation(C_{t})
    let c_t = af::add(&af::mul(&i_t, &ct_t, false).unwrap()
                      , &af::mul(&f_t, &c_tm1, false).unwrap()
                      , false).unwrap();
    let h_t = af::mul(&o_t, &activations::get_activation(outer_activation, &c_t).unwrap(), false).unwrap();

    // store the outputs in the parameter manager
    params.recurrences[LSTMIndex::Input].push(i_t.clone());
    params.recurrences[LSTMIndex::Forget].push(f_t.clone());
    params.recurrences[LSTMIndex::Output].push(o_t.clone());
    params.recurrences[LSTMIndex::CellTilda].push(ct_t.clone());
    params.recurrences[LSTMIndex::Cell].push(c_t.clone()); // TODO: we need c{t-1}
    params.recurrences[LSTMIndex::CellOutput].push(h_t.clone());

    if self.return_sequences {
      Input { data: af::join_many(1, vec![&h_t, &c_t]).unwrap() // join on col
              , activation: self.outer_activation }
    }else {
      Input { data: h_t.clone()
              , activation: self.outer_activation }
    }
  }

  fn backward(&self, params: &mut Params, delta: &Array) -> Array {
    let inner_activation = params.activations[0];
    let outer_activation = params.activations[1];
    let o_t = params.recurrences[LSTMIndex::Output].last().unwrap();
    let c_t = params.recurrences[LSTMIndex::Cell].last().unwrap();

    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    params.deltas = vec![af::mul(delta, &activations::get_activation_derivative(&params.activations[0]
                                                                                , &params.outputs[0].data).unwrap(), false).unwrap()];

    //TODO: redundant?
    // d_h = inner_activation'(z_o)  * outer_activation(c_t) * delta
    let d_h = af::mul(&af::mul(&activations::get_activation_derivative(inner_activation, &o_t).unwrap()
                               , &activations::get_activation(outer_activation, &c_t).unwrap()).unwrap()
                      , delta).unwrap();
    let d_i = self.rtrl(delta, &d_h, &mut params); // input gate delta is returned

    // cleanup members because this layer's backprop is done
    params.recurrences[LSTMIndex::Cell].pop();
    params.recurrences[LSTMIndex::CellOutput].pop();
    params.recurrences[LSTMIndex::CellTilda].pop();
    params.recurrences[LSTMIndex::Forget].pop();
    params.recurrences[LSTMIndex::Input].pop();
    params.recurrences[LSTMIndex::Output].pop();

    let activation_prev = activations::get_activation(self.inputs.activation[0], &self.inputs.data[DataIndex::Input]).unwrap();
    let d_activation_prev = activations::get_activation_derivative(self.inputs.activation[0], &activation_prev).unwrap();
    let delta_prev = af::mul(&af::matmul(&params.weights[0], delta, af::MatProp::TRANS, af::MatProp::NONE).unwrap()
                             , &d_activation_prev, false).unwrap();
    delta_prev
  }
}
