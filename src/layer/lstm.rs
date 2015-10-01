use af;
use af::{Dim4, Array};
use af::MatProp;

use activations;
use initializations;
use layer::{Layer, Input};

pub enum LSTMIndex {
  Input,
  Forget,
  Output,
  CellTilda,
}

pub enum ActivationIndex {
  Input,
  Inner,
  Outer,
}

pub enum DataIndex {
  Input,
  Recurrence,
}

pub struct LSTM {
  weights: Vec<Array>,
  recurrent_weights: Vec<Array>,
  bias: Vec<Array>,
  inner_activation: &'static str,
  outer_activation: &'static str,
  return_sequences: bool,
  delta: Array,
  inputs: Input,
}

impl LSTM {
  pub fn new(input_size: u64, output_size: u64
             , outer_activation: &'static str
             , inner_activation: &'static str
             , w_init: &'static str
             , w_inner_init: &'static str
             , bias_init: &'static str
             , forget_bias_init: &'static str
             , return_sequences: bool) -> LSTM
  {
    LSTM{
      weights: vec![initializations::get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1])).unwrap()],
      recurrent_weights: vec![initializations::get_initialization(w_inner_init, Dim4::new(&[output_size, output_size, 1, 1])).unwrap(); 4],
      bias: vec![initializations::get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()
                 , initializations::get_initialization(bias_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()
                 , initializations::get_initialization(bias_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()
                 , initializations::get_initialization(forget_bias_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()],
      inner_activation: inner_activation,
      outer_activation: outer_activation,
      return_sequences: return_sequences,
    }
  }
}

impl RTRL for LSTM {
  pub fn rtrl(&self, dW_tm1: &mut Array  // previous W derivatives for [I, F, Ct]
              , dU_tm1: &mut Array       // previous U derivatives for [I, F, Ct]
              , db_tm1: &mut Array       // previous b derivatives for [I, F, Ct]
              , z_t: &Array              // current time activation
              , inputs: &Input)          // x_t & h_{t-1}
  {
    let block_size = z_t.dims().unwrap()[0];
    assert!(block_size as f32 % 5.0f32 == 0); // there are 5 data pieces we need
    let chunk_size = block_size / 5;
    // chunk out i, f, ct, c_tm1
    let if_t = activations::get_activation(self.inner_activation,
                                            , &af::rows(z_t, 0, 2*chunk_size).unwrap).unwrap();
    let ct_t = activations::get_activation(self.outer_activation,
                                           , &af::rows(z_t, 3*chunk_size, 4*chunk_size).unwrap).unwrap();
    let c_tm1 = af::rows(z_t, 4*chunk_size, 5*chunk_size).unwrap();

    // diff(z_i), diff(z_f), diff(z_ct)
    let dz = vec![&activations::get_activation_derivative(self.inner_activation, &af::rows(z_t, 0, chunk_size).unwrap()).unwrap() //d(z)
                  , &activations::get_activation_derivative(self.inner_activation, &af::rows(z_t, chunk_size, 2*chunk_size).unwrap()).unwrap()
                  , &activations::get_activation_derivative(self.outer_activation, &af::rows(z_t, 3*chunk_size, 4*chunk_size).unwrap()).unwrap()];
    let ct_ctm1_i = vec![&ct_t, &c_tm1, &af::rows(if_t, 0, chunk_size).unwrap()];

    // [Ct_t; C_{t-1}; i_t] * dz
    let dzprod = af::mul(&af::join_many(0, ct_ctm1_i).unwrap()
                         , af::join_many(0, dzvec).unwrap(), false).unwrap();

    // dC_t/dWi  = (dC_{t-1}/dWi  * f_t) + ct_t  * inner_activation(z_i) * x_t
    // dC_t/dWf  = (dC_{t-1}/dWf  * f_t) + c_tm1 * inner_activation(z_f) * x_t
    // dC_t/dWct = (dC_{t-1}/dWct * f_t) + i_t   * outer_activation(Ct)  * x_t
    let w_lhs = af::mul(dW_tm1, &af::rows(if_t, chunk_size, 2*chunk_size).unwrap(), true).unwrap(); // dC_{t-1}/dW * f_t
    let w_rhs = af::mul(&dzprod, &inputs.data[DataIndex::Input]);
    dW_tm1 = af::add(&w_lhs, &w_rhs).unwrap();

    // dC_t/dUi = (dC_{t-1}/dUi * f_t) + ct_t * inner_activation(z_i) * h_{t-1}
    // dC_t/dUf = (dC_{t-1}/dUf * f_t) + c_tm1 * inner_activation(z_f) * h_{t-1}
    // dC_t/dUct = (dC_{t-1}/dUct * f_t) + outer_activation(Ct) * x_t * h_{t-1}
    let u_lhs = af::mul(dU_tm1, &af::rows(if_t, chunk_size, 2*chunk_size).unwrap(), true).unwrap(); // dC_{t-1}/dU * f_t
    let u_rhs = af::mul(&dzprod, &inputs.data[DataIndex::Recurrence]);
    dU_tm1 = af::add(&u_lhs, &u_rhs).unwrap();

    // dC_t/dbi = (dC_{t-1}/dbi * f_t) + ct_t * inner_activation(z_i)
    // dC_t/dbf = (dC_{t-1}/dbf * f_t) + c_tm1 * inner_activation(z_f)
    // dC_t/dbct = (dC_{t-1}/dbct * f_t) + outer_activation(Ct)
    let b_lhs = af::mul(db_tm1, &af::rows(if_t, chunk_size, 2*chunk_size).unwrap(), true).unwrap(); // dC_{t-1}/db * f_t
    dW_tm1 = af::add(&b_lhs, &dzprod).unwrap();
  }
}

impl Layer for LSTM {

  pub fn forward(&mut self, inputs:& Input) {
    // keep previous layer's outputs
    self.inputs = inputs.clone();

    // apply the activation to the previous layer [Optimization: Memory saving]
    let activated_input = activations::get_activation(inputs.activation[ActivationIndex::Inner]
                                                      , &inputs.data[DataIndex::Input]).unwrap();

    // extract the sub-block of each gate [i_tm1, f_tm1, o_tm1, ct_tm1, c_tm2]
    let block_size = inputs.data[DataIndex::Recurrence].dims().unwrap()[0];
    assert!(block_size as f32 % 5.0f32 == 0); // there are 5 data pieces we need
    let chunk_size = block_size / 5;
    let ifo_tm1 = activations::get_activation(inputs.activation[ActivationIndex::Inner]
                                              , &af::rows(&inputs.data[DataIndex::Recurrence], 0, 3 * chunk_size).unwrap).unwrap();
    let ct_tm1 = activations::get_activation(inputs.activation[ActivationIndex::Outer]
                                             , &af::rows(&inputs.data[DataIndex::Recurrence], 3 * chunk_size, 4 * chunk_size).unwrap).unwrap();
    let c_tm2 = af::rows(&inputs.data[DataIndex::Recurrence], 4 * chunk_size, 5 * chunk_size).unwrap();

    // calculate c_tm1 & h_tm1
    let c_tm1 = af::add(&af::mul(&af::rows(&ifo_tm1, 0, chunk_size).unwrap(), &ct_tm1, false).unwrap()
                        , &af::mul(&af::rows(&ifo_tm1, chunk_size, 2 * chunk_size).unwrap(), &c_tm2, false).unwrap()
                        , false).unwrap();
    let h_tm1 = af::mul(&o_tm1, activations::get_activation(inputs.activation[ActivationIndex::Outer]
                                                            , &c_tm1), false).unwrap();

    // forward pass in a batch for performance
    let weights_ref    = vec![&self.weights[LSTMIndex::Input]
                              , &self.weights[LSTMIndex::Forget]
                              , &self.weights[LSTMIndex::Output]
                              , &self.weights[LSTMIndex::CellTilda]];
    let recurrents_ref = vec![&self.recurrent_weights[LSTMIndex::Input]
                              , &self.recurrent_weights[LSTMIndex::Forget]
                              , &self.recurrent_weights[LSTMIndex::Output]
                              , &self.recurrent_weights[LSTMIndex::CellTilda]];
    let bias_ref       = vec![&self.bias[LSTMIndex::Input]
                              , &self.bias[LSTMIndex::Forget]
                              , &self.bias[LSTMIndex::Output]
                              , &self.bias[LSTMIndex::CellTilda]];
    // [ifo_ct] = W*x + U*h_tm1 + b
    let z_t = af::add(&af::add(&af::matmul(&af::join_many(0, weights_ref).unwrap(), &activated_input).unwrap()
                               , &af::matmul(&af::join_many(0, recurrents_ref).unwrap(), &h_tm1).unwrap(), false).unwrap()
                      , &af::join_many(0, bias_ref).unwrap(), true).unwrap();
    self.rtrl(&self.dW, &self.dU, &self.db, &z_t, inputs);
    // since we are RTRL'ing I, F, Ct w.r.t. C we just need to shoot out O
    if self.return_sequences {
      Input { data: af::join_many(0, vec![&z_t, &c_tm1]).unwrap()
              , activation: vec![self.inner_activation, self.outer_activation] }
    }else { //TODO: Fix this
      Input { data: af::join_many(0, vec![&ifo_tm1, &ct_tm1, &c_tm1]).unwrap()
              , activation: vec![self.inner_activation, self.outer_activation] }
    }
  }

  fn backward(&mut self, delta: &Array) -> Array {
  self.delta = delta.clone();

  let activation_prev = activations::get_activation(self.inputs.activation[0], &self.inputs.data[DataIndex::Input]).unwrap();
  let d_activation_prev = activations::get_activation_derivative(self.inputs.activation[0], &activation_prev).unwrap();
  let delta_prev = af::mul(&af::matmul(&self.weights[0], delta, af::MatProp::TRANS, af::MatProp::NONE).unwrap()
                           , &d_activation_prev, false).unwrap();
  delta_prev
}

fn get_delta(&self) -> Array {
self.delta.clone()
  }

  fn get_weights(&self) -> Vec<Array> {
    self.recurrent_weights.extend(self.weights.iter().cloned()).clone()
  }

  fn set_weights(&mut self, weights: &Array, index: usize) {
    self.weights[index] = weights.clone();
  }

  fn get_bias(&self) -> Vec<Array> {
    self.bias.clone()
  }

  fn set_bias(&mut self, bias: &Array, index: usize) {
    self.bias[index] = bias.clone();
  }

  fn get_bias_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for b in &self.bias {
      dims.push(b.dims().unwrap().clone())
    }
    dims
  }

  fn get_weight_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for w in &self.weights {
      dims.push(w.dims().unwrap().clone())
    }
    for w in &self.recurrent_weights {
      dims.push(w.dims().unwrap().clone())
    }
    dims
  }

  fn get_input(&self) -> Input {
    self.inputs.clone()
  }

  fn output_size(&self) -> u64 {
    let weight_dims = self.get_weight_dims();
    weight_dims[weight_dims.len() - 1][1]
  }

  fn input_size(&self) -> u64 {
    let weight_dims = self.get_weight_dims();
    weight_dims[0][0]
  }

  fn get_activation_type(&self) -> &'static str {
    &self.activation
  }
}
