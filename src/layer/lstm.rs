use af;
use af::{Dim4, Array};
use af::MatProp;

use activations;
use initializations;
use layer::{Layer, Input};

pub enum LSTMWeightIndex {
  Input,
  Output,
  Forget,
  Cell,
}

pub struct LSTM {
  weights: Vec<Array>,
  bias: Vec<Array>,
  inner_activation: &'static str,
  outer_activation: &'static str,
  return_sequences: bool,
  delta: Array,
  inputs: Input,
  indexer: af::Indexer,
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
    weights: vec![initializations::get_initialization(w_inner_init, Dim4::new(&[output_size, input_size, 1, 1])).unwrap()],
    recurrent_weights: vec![initializations::get_initialization(w_inner_init, Dim4::new(&[output_size, output_size, 1, 1])).unwrap(); 4],
    bias: vec![initializations::get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()
               , initializations::get_initialization(bias_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()
               , initializations::get_initialization(bias_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()
               , initializations::get_initialization(forget_bias_init, Dim4::new(&[output_size, 1, 1, 1])).unwrap()],
    indexer: af::Indexer::new().unwrap(),
    inner_activation: inner_activation,
    outer_activation: outer_activation,
    return_sequences: return_sequences,
  }

  pub fn split_input(&self, x: &Array) -> Vec<Array> {
    //let retval = Vec::new();
    self.indexer.set_index();
    af::assign_gen(x, 
  }
}

impl Layer for LSTM {

  pub fn forward(&mut self, input:& Input) {
    // keep previous layer's outputs
    self.inputs = initializations.clone();

    // apply the activation to the previous layer [Optimization: Memory saving]
    let activated_input = activations::get_activation(input.activation, &input.data).unwrap();

    // forward pass on input date
    let input = af::add(&af::matmul(&self.weights[LSTMWeightIndex::Input]
                                    , &activated_input
                                    , MatProp::NONE
                                    , MatProp::NONE).unwrap()
                        , &self.bias[LSTMWeightIndex::Input]).unwrap();

    // forward pass on the forget gate
    let forget = af::add(&af::matmul(&self.weights[LSTMWeightIndex::Forget]
                                     , &activated_input
                                     , MatProp::NONE
                                     , MatProp::NONE).unwrap()
                         , &self.bias[LSTMWeightIndex::Forget]).unwrap();

    // forward pass on the cell gate
    let cell = af::add(&af::matmul(&self.weights[LSTMWeightIndex::Cell]
                                   , &activated_input
                                   , MatProp::NONE
                                   , MatProp::NONE).unwrap()
                       , &self.bias[LSTMWeightIndex::Cell]).unwrap();

    // forward pass on the output gate
    let output = af::add(&af::matmul(&self.weights[LSTMWeightIndex::Output]
                                     , &activated_input
                                     , MatProp::NONE
                                     , MatProp::NONE).unwrap()
                         , &self.bias[LSTMWeightIndex::Output]).unwrap();

    Input {data: af::add(&mul, &self.bias[0], true).unwrap(), activation: self.activation}
  }


}

