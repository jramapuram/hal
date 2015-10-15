use af;
use af::{Dim4, Array};
use af::MatProp;

use activations;
use initializations;
use params::{LSTMIndex, Input, Params};

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
  pub fn rtrl(&self, dW_tm1: &mut Array  // previous W derivatives for [I, F, Ct]
              , dU_tm1: &mut Array       // previous U derivatives for [I, F, Ct]
              , db_tm1: &mut Array       // previous b derivatives for [I, F, Ct]
              , z_t: &Array              // current time activation
              , inputs: &Input           // x_t
              , recurrences: &Input      // h_{t-1}
  {
    let block_size = z_t.dims().unwrap()[0]; // the batch size * 5
    assert!(block_size as f32 % 5.0f32 == 0); // there are 5 data pieces we need
    let chunk_size = block_size / 5;
    // chunk out zi, zf, zct, zc_tm1
    let zi_t = af::rows(z_t, 0, chunk_size).unwrap();
    let zf_t = af::rows(z_t, chunk_size, 2*chunk_size).unwrap();
    let zct_t = af::rows(z_t, 3*chunk_size, 4*chunk_size).unwrap();

    // compute their activations
    let i_t =  activations::get_activation(self.inner_activation, &zi_t).unwrap();
    let f_t =  activations::get_activation(self.inner_activation, &zf_t).unwrap();
    let ct_t = activations::get_activation(self.outer_activation, &zct_t).unwrap();
    let c_tm1 = af::rows(z_t, 4*chunk_size, 5*chunk_size).unwrap();

    // compute their derivatives [diff(z_i), diff(z_f), diff(z_ct)]
    let dz = vec![&activations::get_activation_derivative(self.inner_activation, &zi_t).unwrap()
                  , &activations::get_activation_derivative(self.inner_activation, &zf_t).unwrap()
                  , &activations::get_activation_derivative(self.outer_activation, &zct_t).unwrap()];
    let ct_ctm1_it = vec![&ct_t, &c_tm1, &i_t).unwrap()];

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
    dW_tm1 = af::add(&b_lhs, &dzprod, false).unwrap();
  }
}

impl Layer for LSTM {
  fn forward(&self, params: &mut Params
             , inputs: &Input
             , recurrence: &Option<Input>) -> (Input, Option<Input>)
  {
    // inputs = x_tm1
    // recurrence = h_tm1

    // keep previous layer's outputs
    assert!(inputs.data.dims().unwrap()[2] == 1);
    params.inputs[0] = vec![inputs.clone()];

    // apply the activation to the previous layer [Optimization: Memory saving]
    // the input activation is the activation of the previous output [outer]
    // let activated_input = activations::get_activation(inputs.activation        // self.inputs.activation[ActivationIndex::Inner]
    //                                                   , inputs.data).unwrap(); //&self.inputs.data[DataIndex::Input]).unwrap();

    let inner_activation = params.activations[0];
    let outer_activation = params.activations[1];

    let i_tm1;   // input gate @ t-1
    let f_tm1;   // forget gate @ t-1
    let o_tm1;   // output gate @ t-1
    let ct_tm1;  // cell internal @ t-1
    let c_tm2;   // cell output @ t-2

    if recurrence.is_some() {
      // extract the sub-block of each gate [i_tm1, f_tm1, o_tm1, ct_tm1, c_tm2]
      let block_size = recurrence.data.dims().unwrap()[0];
      assert!(block_size as f32 % 5.0f32 == 0);
      let chunk_size = block_size / 5;

      // i_{t-1} = inner_activation(zi_{t-1})
      // f_{t-1} = inner_activation(zf_{t-1})
      // o_{t-1} = inner_activation(zo_{t-1})
      ifo_tm1 = activations::get_activation(recurrence.activation, &af::rows(&recurrence.data, 0, 3 * chunk_size).unwrap()).unwrap();

      // Ct_{t-1} = outer_activation(zct_{t-1})
      ct_tm1 = activations::get_activation(inputs.activation
                                           , &af::rows(&recurrence.data , 3 * chunk_size, 4 * chunk_size).unwrap()).unwrap();
      // C_{t-2} = last_chunk(recurrence)
      c_tm2 = af::rows(&recurrence.data, 4 * chunk_size, 5 * chunk_size).unwrap();
    }else { // this is the first node in the recurrence
      let batch_size = inputs.data.dims().unwrap()[0];
      ifo_recurrence_dims = Dim4::new([3*batch_size, self.output_size, 1, 1]).unwrap();
      ct_recurrence_dims = Dim4::new([batch_size, self.output_size, 1, 1]).unwrap();
      c_recurrence_dims = Dim4::new([batch_size, self.output_size, 1, 1]).unwrap();
      ifo_tm1 = initializations::get_initialization("zeros", ifo_recurrence_dims).unwrap();
      ct_tm1 = initializations::get_initialization("zeros", ct_recurrence_dims).unwrap();
      c_tm2 = initializations::get_initialization("zeros", c_recurrence_dims).unwrap();
    }

    // extract these gate values
    let i_tm1 = af::rows(&ifo_tm1, 0, chunk_size).unwrap();
    let f_tm1 = af::rows(&ifo_tm1, chunk_size, 2*chunk_size).unwrap();
    let o_tm1 = af::rows(&ifo_tm1, 2*chunk_size, 3*chunk_size).unwrap();

    // C_{t-1} = i_{t-1} * Ct_{t-1} + f_{t-1} * C_{t-2}
    // h_{t-1} = o_{t-1} * outer_activation(C_{t-1})
    let c_tm1 = af::add(&af::mul(&i_tm1, &ct_tm1, false).unwrap()
                        , &af::mul(&f_tm1, &c_tm2, false).unwrap()
                        , false).unwrap();
    let h_tm1 = af::mul(&o_tm1, activations::get_activation(inputs.activation, &c_tm1).unwrap(), false).unwrap();

    // forward pass in a batch for performance
    let weights_ref    = vec![&params.weights[LSTMIndex::Input]
                              , &params.weights[LSTMIndex::Forget]
                              , &params.weights[LSTMIndex::Output]
                              , &params.weights[LSTMIndex::CellTilda]];
    let offset = 3; // the offset from weights --> recurrent weights
    let recurrents_ref = vec![&params.weights[LSTMIndex::Input as usize + offset]
                              , &params.weights[LSTMIndex::Forget as usize + offset]
                              , &params.weights[LSTMIndex::Output as usize + offset]
                              , &params.weights[LSTMIndex::CellTilda as usize + offset]];
    let bias_ref       = vec![&params.biases[LSTMIndex::Input]
                              , &params.biases[LSTMIndex::Forget]
                              , &params.biases[LSTMIndex::Output]
                              , &params.biases[LSTMIndex::CellTilda]];
    // [z(i,f,o,ct)_t] = W*x + U*h_tm1 + b
    let z_t = af::add(&af::add(&af::matmul(&af::join_many(0, weights_ref).unwrap(), &activated_input).unwrap()
                               , &af::matmul(&af::join_many(0, recurrents_ref).unwrap(), &h_tm1).unwrap(), false).unwrap()
                      , &af::join_many(0, bias_ref).unwrap(), true).unwrap();

    // since we are RTRL'ing i, f, Ct w.r.t. C we technically just need to pass z_o & c_t
    if self.return_sequences {
      (Input { data: af::join_many(0, vec![&z_t, &c_tm1]).unwrap()
              , activation: self.inner_activation, self.outer_activation] }
    }else {
      Input { data: vec![z_t.clone()]
              , activation: vec![self.inner_activation, self.outer_activation] }
    }
  }

  fn backward(&self, params: &mut Params, delta: &Array) -> Array{
    self.delta = delta.clone();
    let inner_activation = params.activations[0];
    let outer_activation = params.activations[1];

    // d_h = inner_activation'(z_o)  * outer_activation(c_t) * delta
    let d_h = af::mul(&af::mul(&activations::get_activation_derivative(inner_activation, params.recurrences[LSTMIndex::Output]).unwrap()
                               , &activations::get_activation(outer_activation, params.recurrences[LSTMIndex::Cell]).unwrap()).unwrap()
                      , delta).unwrap();
    // e_t = o_t * outer_activation'(c_t) * delta
    let e_t = af::mul(&af::mul(&activations::get_activation(inner_activation, params.recurrences[LSTMIndex::Output]).unwrap()
                               , &activations::get_activation_derivative(outer_activation, params.recurrences[LSTMIndex::Cell]).unwrap()).unwrap()
                      , delta).unwrap();
    // dW = delta[0]
    // dU = delta[1]
    // db = delta[2]
    //self.rtrl(&self.dW, &self.dU, &self.db, &z_t, inputs);
    self.rtrl(, &self.dU, &self.db, &z_t, inputs);

    let activation_prev = activations::get_activation(self.inputs.activation[0], &self.inputs.data[DataIndex::Input]).unwrap();
    let d_activation_prev = activations::get_activation_derivative(self.inputs.activation[0], &activation_prev).unwrap();
    let delta_prev = af::mul(&af::matmul(&params.weights[0], delta, af::MatProp::TRANS, af::MatProp::NONE).unwrap()
                             , &d_activation_prev, false).unwrap();
    delta_prev
  }
}
