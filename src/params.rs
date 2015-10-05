use af;
use af::{Array, Dim4};
use std::default::Default;
use itertools::Zip;

use activations;
use initializations;
use error::HALError;

#[derive(Clone)]
pub struct Input {
  pub data: Array,
  pub activation: &'static str,
}

impl Default for Input {
  fn default() -> Input {
    Input {
      data: initializations::empty(),
      activation: "ones",
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
  inputs: Vec<Input>,
}

pub struct ParamManager {
  layer_storage: Vec<Params>,
  has_recurrence: bool,
}

impl Default for ParamManager {
  fn default() -> ParamManager {
    ParamManager {
      layer_storage: Vec::new(),
      has_recurrence: false,
    }
  }
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
    for (b_init, b_dims) in Zip::new((&biases_init, &biases_dims)) {
      biases.push(self.generate(b_init, b_dims));
    }

    if layer_type == "lstm" || layer_type == "gru" || layer_type == "rnn"{
      self.has_recurrence = true;
    }

    self.layer_storage.push(Params{
      layer_type: layer_type,
      weights: weights,
      biases: biases,
      activations: activations,
      delta: Vec::new(),
      inputs: Vec::new(),
    });
  }

  fn generate(init: &'static str, dims: (usize, usize)) -> Array {
    let dims = Dim4::new(&[dims.0, dims.1, 1, 1]);
    initializations::get_initialization(init, dims).unwrap()
  }

  fn get_params(&self, layer_index: usize) -> Params {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].clone()
  }

  fn get_weights(&self, layer_index: usize) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].weights.clone()
  }

  fn get_weight(&self, layer_index: usize, weight_num: usize) -> Array {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].weights.len() - 1 <= weight_num);
    self.layer_storage[layer_index].weights[weight_num].clone()
  }

  fn get_biases(&self, layer_index: usize) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].bias.clone()
  }

  fn get_bias(&self, layer_index: usize, bias_num: usize) -> Array {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].biases.len() - 1 <= bias_num);
    self.layer_storage[layer_index].biases[bias_num].clone()
  }

  fn get_bias_dims(&self, layer_index: usize) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for b in &self.layer_storage[layer_index].biases {
      dims.push(b.dims().unwrap().clone());
    }
    dims
  }

  fn get_weight_dims(&self, layer_index: usize) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for w in &self.layer_storage[layer_index].weights {
      dims.push(w.dims().unwrap().clone());
    }
    dims
  }

  fn set_weights(&mut self, layer_index: usize, weights: Vec<Array>){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].weights = weights;
  }

  fn set_weight(&mut self, layer_index: usize, weight_num: usize, weight: Array){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].weights.len() - 1 <= weight_num);
    self.layer_storage[layer_index].weights[weight_num] = weight;
  }

  fn set_biases(&self, layer_index: usize, biases: Vec<Array>){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].biases = biases;
  }

  fn set_bias(&mut self, layer_index: usize, bias_num: usize, bias: Array){
    assert!(self.layer_storage.len() - 1 <= layer_index);
    assert!(self.layer_storage[layer_index].bias.len() - 1 <= bias_num);
    self.layer_storage[layer_index].biases[bias_num] = bias;
  }

  fn get_delta(&self, layer_index: usize) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].delta.clone()
  }

  fn set_delta(&mut self, layer_index: usize, delta: Vec<Array>) {
    assert!(self.layer_storage.len() - 1 <= layer_index);
    self.layer_storage[layer_index].delta = delta;
  }

  fn tie_weight(&self, layer_input: usize, iweight_index: usize
                , layer_output: usize, oweight_index: usize)
  {
    assert!(self.layer_storage.len() - 1 <= layer_input);
    assert!(self.layer_storage.len() - 1 <= layer_output);
    assert!(self.layer_storage[layer_input].weights.len() - 1 <= iweight_index);
    assert!(self.layer_storage[layer_output].weights.len() - 1 <= oweight_index);
    let input_dims = self.layer_storage[layer_input].weights[iweight_index].dims().unwrap();
    let output_dims = self.layer_storage[layer_output].weights[oweight_index].dims().unwrap();
    assert!(input_dims == output_dims
            || (input_dims[0] == output_dims[1]
                && input_dims[1] == output_dims[0]));
    self.layer_storage[layer_output].weights[oweight_index]
      = self.layer_storage[layer_input].weights[iweight_index].clone();
  }

  fn tie_bias(&self, layer_input: usize, ibias_index: usize
              , layer_output: usize, obias_index: usize)
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
               , input_size: usize
               , output_size: usize
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
              , input_size: usize
              , output_size: usize
              , input_activation: &'static str
              , output_activation: &'static str
              , w_inner_init: &'static str
              , w_outer_init: &'static str
              , forget_bias_init: &'static str
              , b_init: &'static str);
  fn get_recurrences(&self, layer_index: usize) -> Vec<Array>;
  fn get_recurrence(&self, layer_index: usize, recur_name: LSTMIndex) -> Array;
  fn set_recurrences(&self, layer_index: usize, recurrences: Vec<Array>);
  fn set_recurrence(&self, layer_index: usize, recur_name: LSTMIndex, recurrence: Array);
}

/** Custom Layer Impls **/
impl DenseGenerator for ParamManager {
  fn add_dense(&mut self
               , input_size: usize
               , output_size: usize
               , activation: &'static str
               , w_init: &'static str
               , b_init: &'static str)
  {
    self.add("dense"
             , vec![w_init]
             , vec![(output_size, input_size)]
             , vec![b_init]
             , vec![(output_size, 1)]
             , vec![activation]);
  }
}

impl LSTMGenerator for ParamManager {
  fn add_lstm(&mut self
              , input_size: usize
              , output_size: usize
              , inner_activation: &'static str
              , outer_activation: &'static str
              , w_init: &'static str
              , w_recurrent_init: &'static str
              , forget_b_init: &'static str
              , b_init: &'static str)
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
             , vec![b_init, forget_b_init, b_init, b_init]
             , vec![bias_dims; 4]
             , vec![inner_activation, outer_activation]);
  }

  fn get_recurrences(&self, layer_index: usize) -> Vec<Array>{
    let offset = 4;
    vec![self.get_weight(layer_index, LSTMIndex::Input + offset)
         , self.get_weight(layer_index, LSTMIndex::Forget + offset)
         , self.get_weight(layer_index, LSTMIndex::Output + offset)
         , self.get_weight(layer_index, LSTMIndex::CellTilda + offset)]
  }

  fn get_recurrence(&self, layer_index: usize, recur_name: LSTMIndex) -> Array{
    let offset = 4;
    self.get_weight(layer_index, recur_name + offset)
  }

  fn set_recurrences(&self, layer_index: usize, recurrences: Vec<Array>){
    self.set_recurrence(layer_index, LSTMIndex::Input, recurrences[LSTMIndex::Input]);
    self.set_recurrence(layer_index, LSTMIndex::Forget, recurrences[LSTMIndex::Forget]);
    self.set_recurrence(layer_index, LSTMIndex::Output, recurrences[LSTMIndex::Output]);
    self.set_recurrence(layer_index, LSTMIndex::CellTilda, recurrences[LSTMIndex::CellTilda]);
  }

  fn set_recurrence(&self, layer_index: usize, recur_name: LSTMIndex, recurrence: Array){
    let offset = 4;
    self.set_weight(layer_index, offset + recur_name, recurrence);
  }
}
