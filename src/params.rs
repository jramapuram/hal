use af;
use af::{Array, Dim4};
use std::default::Default;

use layer::Input;
use activations;
use initializations;
use error::HALError;

#[derive(Clone)]
pub struct Input {
  pub data: Vec<Array>,
  pub activation: Vec<&'static str>,
}

impl Default for Input {
  fn default() -> Input {
    Input {
      data: Vec::new(),
      activation: Vec::new(),
    }
  }
}

#[derive(Clone)]
pub struct Params {
  layer_type: &'static str,
  weights: Vec<Array>,
  biases: Vec<Array>,
  activations: Vec<&'static str>,
  delta: Vec<Array>,
  inputs: Input,
}

pub struct ParamManager {
  layer_storage: Vec<Params>,
  has_recurrence: bool,
}

impl ParamManager {
  fn add(&mut self
         , layer_type: &'static str
         , weight_init: Vec<&'static str>
         , weight_dims: Vec<(u64, u64)>
         , biases_init: Vec<&'static str>
         , biases_dims: Vec<(u64, u64)>
         , activations: Vec<&'static str>)
  {
    // generate the weights
    let mut weights: Vec<(Array, &'static str)> = Vec::with_capacity(weight_dims.len());
    for (w_init, w_dims) in Zip::new((&weight_init, &weight_dims)) {
      weights.push(self.generate(w_init, w_dims));
    }
    // generate the biases
    let mut biases: Vec<(Array, &'static str)> = Vec::with_capacity(weight_dims.len());
    for (b_init, b_dims) in Zip::new((&bias_init, &bias_dims)) {
      biases.push(self.generate(b_init, b_dims));
    }

    if layer_type == "lstm" || layer_type == "gru" || layer_type == "rnn"{
      self.has_recurrence = true;
    }

    self.layer_storage.push(Params{
      layer_type: layer_type
      weights: weights,
      biases: biases,
      activations: activations,
      delta: Vec::new(),
      inputs: Input::default()
    });
  }

  fn generate(init: &'static str, dims: (u64, u64)) -> Array {
    let dims = Dim4::new(&[dims.0, dims.1, 1, 1]);
    initializations::get_initialization(w_init, dims).unwrap()
  }

  fn get_weights(&self, layer_index: u64) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].weights.clone()
  }

  fn get_weight(&self, layer_index: u64, weight_num: u64) -> Array {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].weights.len() - 1 <= weight_num);
    self.layer_storage[layer_index].weights[weight_num].clone()
  }

  fn get_biases(&self, layer_index: u64) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].bias.clone()
  }

  fn get_bias(&self, layer_index: u64, bias_num: u64) -> Array {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].biases.len() - 1 <= bias_num);
    self.layer_storage[layer_index].biases[bias_num].clone()
  }

  fn set_weights(&mut self, layer_index: u64, weights: &Vec<Array>){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].weights = weights.clone();
  }

  fn set_weight(&mut self, layer_index: u64, weight_num: u64, weight: Array){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].weights.len() - 1 <= weight_num);
    self.layer_storage[layer_index].weights[weight_num] = weight;
  }

  fn set_biases(&self, layer_index: u64, biases: &Vec<Array>){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].biases = biases.clone();
  }

  fn set_bias(&mut self, layer_index: u64, bias_num: u64, bias: Array){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].bias.len() - 1 <= bias_num);
    self.layer_storage[layer_index].biases[bias_num] = bias;
  }

  fn tie_weight(&self, layer_input: u64, iweight_index: u64
                , layer_output: u64, oweight_index: u64)
  {
    assert!(self.layer_storage.len() - 1 <= layer_input);
    assert!(self.layer_storage.len() - 1 <= layer_output);
    assert!(self.layer_storage.[layer_input].weights.len() - 1 <= iweight_index);
    assert!(self.layer_storage.[layer_output].weights.len() - 1 <= oweight_index);
    let input_dims = self.layer_storage[layer_input].weights[iweight_index].dims().unwrap();
    let output_dims = self.layer_storage[layer_output].weights[oweight_index].dims().unwrap();
    assert!(input_dims == output_dims
            || (input_dims[0] == output_dims[1]
                && input_dims[1] == output_dims[0]));
    self.layer_storage[layer_output].weights[oweight_index]
      = self.layer_storage[layer_input].weights[iweight_index].clone();
  }

  fn tie_bias(&self, layer_input: u64, ibias_index: u64
              , layer_output: u64, obias_index: u64)
  {
    assert!(self.layer_storage.len() - 1 <= layer_input);
    assert!(self.layer_storage.len() - 1 <= layer_output);
    assert!(self.layer_storage[layer_input].biases.len() - 1 <= ibias_index);
    assert!(self.layer_storage[layer_output].biases.len() - 1 <= obias_index);
    let input_dims = self.layer_storage[layer_input].biases[ibias_index].dims().unwrap();
    let output_dims = self.layer_storage[layer_output].biases[obias_index].dims().unwrap();
    assert!(input_dims == output_dims);
    self.layer_storage[layer_output].biases[obias_index]
      = self.layer_storage[layer_input].biases[ibias_index].clone();
  }
}

/** Custom Layer Traits **/
pub trait DenseGenerator {
  fn add_dense(&mut self
               , input_size: u64
               , output_size: u64
               , activation: &'static str
               , w_init: &'static str
               , b_init: &'static str);
}

pub enum LSTMIndex {
  Input,
  Forget,
  Output,
  CellTilda,
}

pub trait LSTMGenerator {
  fn add_lstm(&mut self
              , input_size: u64
              , output_size: u64
              , input_activation: &'static str
              , output_activation: &'static str
              , w_inner_init: &'static str
              , w_outer_init: &'static str
              , forget_bias_init: &'static str
              , b_init: &'static str);
  fn get_recurrences(&self, layer_index: u64) -> Vec<Array>;
  fn get_recurrence(&self, layer_index: u64, recur_name: LSTMIndex) -> Array;
  fn set_recurrences(&self, layer_index: u64, recurrences: &Vec<Array>);
  fn set_recurrence(&self, layer_index: u64, recur_name: LSTMIndex, recurrence: &Array);
}

/** Custom Layer Impls **/
impl DenseGenerator for ParamManager {
  fn add_dense(&mut self
               , input_size: u64
               , output_size: u64
               , activation: &'static str
               , w_init: &'static str
               , b_init: &'static str)
  {
    self.add("dense"
             , vec![w_init]
             , vec![(output_size, input_size)]
             , vec![bias_init]
             , vec![(output_size, 1)]
             , vec![activations]);
  }
}

impl LSTMGenerator for ParamManager {
  fn add_lstm(&mut self
              , input_size: u64
              , output_size: u64
              , inner_activation: &'static str
              , outer_activation: &'static str
              , w_init: &'static str
              , w_recurrent_init: &'static str
              , forget_bias_init: &'static str
              , bias_init: &'static str)
  {
    let input_dims = (output_size, input_size);
    let recurrent_dims = (output_size, output_size);
    let bias_dims = (output_size, 1);
    // W_i, W_f, W_o, W_ct, U_i, U_f, U_o, U_ct
    self.add("lstm"
             , vec![w_init, w_init, w_init, w_init
                    , w_recurrent_init, w_recurrent_init, w_recurrent_init, w_recurrent_init]
             , vec![input_dims, input_dims, input_dims, input_dims
                    , recurrent_dims, recurrent_dims, recurrent_dims, recurrent_dims]
             , vec![bias_init, forget_bias_init, bias_init, bias_init]
             , vec![bias_dims; 4]
             , vec![inner_activation, outer_activation]);
  }

  fn get_recurrences(&self, layer_index: u64) -> Vec<Array>{
    let offset = 4;
    vec![self.get_weight(layer_index, LSTMIndex::Input + offset)
         , self.get_weight(layer_index, LSTMIndex::Forget + offset)
         , self.get_weight(layer_index, LSTMIndex::Output + offset)
         , self.get_weight(layer_index, LSTMIndex::CellTilda + offset)]
  }

  fn get_recurrence(&self, layer_index: u64, recur_name: LSTMIndex) -> Array{
    let offset = 4;
    self.get_weight(layer_index, recur_name + offset)
  }

  fn set_recurrences(&self, layer_index: u64, recurrences: &Vec<Array>){
    self.set_recurrence(layer_index, LSTMIndex::Input, recurrences[LSTMIndex::Input]);
    self.set_recurrence(layer_index, LSTMIndex::Forget, recurrences[LSTMIndex::Forget]);
    self.set_recurrence(layer_index, LSTMIndex::Output, recurrences[LSTMIndex::Output]);
    self.set_recurrence(layer_index, LSTMIndex::CellTilda, recurrences[LSTMIndex::CellTilda]);
  }

  fn set_recurrence(&self, layer_index: u64, recur_name: LSTMIndex, recurrence: Array){
    let offset = 4;
    self.set_weight(layer_index, offset + recur_name, recurrence);
  }
}
