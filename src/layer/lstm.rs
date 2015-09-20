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
  Inner,
  Outer,
}

pub enum DataIndex {
  Input,
  Recurrence,
}

pub struct LSTM {
  weights: Vec<Array>,
  recurrent_weights: Vec<Array>
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

impl Layer for LSTM {

  pub fn forward(&mut self, inputs:& Input) {
    // keep previous layer's outputs
    self.inputs = inputs.clone();

    // apply the activation to the previous layer [Optimization: Memory saving]
    let activated_input = activations::get_activation(inputs.activation[DataIndex::Input], &inputs.data[DataIndex::Input]).unwrap();

    // extract the sub-block of each gate [i_tm1, f_tm1, o_tm1, ct_tm1, c_tm2]
    let block_size = inputs.data[DataIndex::Recurrence].dims().unwrap()[0];
    assert!(block_size as f32 % 5.0f32 == 0);
    let chunk_size = block_size / 5;
    let i_f_o_tm1 = activations::get_activation(inputs.activation[ActivationIndex::Inner]
                                                , &af::rows(&inputs.data[DataIndex::Recurrence], 0, 3 * chunk_size).unwrap).unwrap();
    let ct_tm1 = activations::get_activation(inputs.activation[ActivationIndex::Outer]
                                             , &af::rows(&inputs.data[DataIndex::Recurrence], 3 * chunk_size, 4 * chunk_size).unwrap).unwrap();
    let c_tm2 = af::rows(&inputs.data[DataIndex::Recurrence], 4 * chunk_size, 5 * chunk_size).unwrap();

    // calculate c_tm1 & h_tm1
    let c_tm1 = af::add(&af::mul(&af::rows(&i_f_o_tm1, 0, chunk_size).unwrap(), &ct_tm1, false).unwrap()
                        , &af::mul(&af::rows(&i_f_o_tm1, chunk_size, 2 * chunk_size).unwrap(), &c_tm2, false).unwrap()
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
    let z_t = af::add(&af::add(&af::matmul(&af::join_many(0, weights_ref).unwrap(), &activated_input).unwrap()
                               , &af::matmul(&af::join_many(0, recurrents_ref).unwrap(), &h_tm1).unwrap(), false).unwrap()
                      , &af::join_many(0, bias_ref).unwrap(), true).unwrap();

    Input { data: af::join_many(0, vec![&i_f_o_tm1, &ct_tm1, &c_tm1]).unwrap(), activation: vec![self.inner_activation, self.outer_activation] }
  }

  fn backward(&mut self, delta: &Array) -> Array {
    // d_l = (transpose(W) * d_{l}) .* dActivation(z-1) where z = activation w/out non-linearity
    self.delta = delta.clone();
    let activation_prev = activations::get_activation(self.inputs.activation[0], &self.inputs.data[0]).unwrap();
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
