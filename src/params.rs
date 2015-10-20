use af::{Array, Dim4};
use std::default::Default;
use itertools::Zip;

use initializations;
//use error::HAL Error;

macro_rules! set_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize, p: Vec<$base_type>) {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      self.layer_storage[layer_index].$vec_extension = p;
    }
    )
}

macro_rules! get_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize) -> Vec<$base_type> {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      self.layer_storage[layer_index].$vec_extension.clone()
    }
    )
}

macro_rules! get_mut_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize) -> &mut Vec<$base_type> {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      &mut self.layer_storage[layer_index].$vec_extension
    }
    )
}

macro_rules! set_param_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize, num: usize, p: $base_type) {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      assert!(self.layer_storage[layer_index].$vec_extension.len() - 1 >= num);
      self.layer_storage[layer_index].$vec_extension[num] = p;
    }
    )
}

macro_rules! get_param_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize, num: usize) -> $base_type {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      assert!(self.layer_storage[layer_index].$vec_extension.len() - 1 >= num);
      self.layer_storage[layer_index].$vec_extension[num].clone()
    }
    )
}

macro_rules! get_mut_param_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize, num: usize) -> &mut $base_type {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      assert!(self.layer_storage[layer_index].$vec_extension.len() - 1 >= num);
      &mut self.layer_storage[layer_index].$vec_extension[num]
    }
    )
}

#[derive(Clone)]
pub struct Input {
  pub data: Array,
  pub activation: String,
}

#[derive(Clone)]
pub struct Params {
  pub layer_type: String,
  pub weights: Vec<Array>,
  pub biases: Vec<Array>,
  pub activations: Vec<String>,
  pub deltas: Vec<Vec<Array>>,
  pub inputs: Vec<Input>,
  pub outputs: Vec<Input>,
  pub recurrences: Vec<Vec<Array>>,
  pub optional: Vec<Vec<Array>>,
}

pub struct ParamManager {
  layer_storage: Vec<Params>,
}

impl Default for ParamManager {
  fn default() -> ParamManager {
    ParamManager {
      layer_storage: Vec::new(),
    }
  }
}

impl ParamManager {
  pub fn add(&mut self
             , layer_type: &str
             , weight_params: Vec<(&str, (usize, usize))> //(init, (i, o))
             , biases_params: Vec<(&str, (usize, usize))> //(init, (i, o))
             , activations: Vec<&str>
             , recurrence_dims: Option<Vec<(usize, usize)>>
             , optional: Option<Vec<(usize, usize)>>)
  {
    // generate the weights
    let mut weights: Vec<Array> = Vec::with_capacity(weight_params.len());
    for (w_init, w_dims) in weight_params {
      weights.push(self.generate(w_init, w_dims));
    }
    // generate the biases
    let mut biases: Vec<Array> = Vec::with_capacity(biases_params.len());
    for (b_init, b_dims) in biases_params {
      biases.push(self.generate(b_init, b_dims));
    }

    // generate recurrence vectors
    let mut recurrences: Vec<Array> = Vec::new();
    for r_dims in recurrence_dims {
      recurrences.push(self.generate("zeros", r_dims));
    }

    // some elements have optional params
    let mut optional: Vec<Array> = Vec::new();
    for o_dims in optional {
      optional.push(self.generate("zeros", r_dims));
    }

    let owned_activations = activations.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    self.layer_storage.push(Params{
      layer_type: layer_type.to_string(),
      weights: weights,
      biases: biases,
      activations: owned_activations,
      deltas: Vec::new(),
      inputs: Vec::new(),
      outputs: Vec::new(),
      recurrences: vec![recurrences],
      optional: vec![optional],
    });
  }

  fn generate(&self, init: &str, dims: &(usize, usize)) -> Array {
    let dims = Dim4::new(&[dims.0 as u64, dims.1 as u64, 1, 1]);
    initializations::get_initialization(init, dims).unwrap()
  }

  pub fn num_layers(&self) -> usize {
    self.layer_storage.len()
  }

  pub fn get_params(&self, layer_index: usize) -> Params {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].clone()
  }

  pub fn get_mut_params(&mut self, layer_index: usize) -> &mut Params {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    &mut self.layer_storage[layer_index]
  }

  pub fn get_weights(&self, layer_index: usize) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].weights.clone()
  }

  pub fn get_weight(&self, layer_index: usize, weight_num: usize) -> Array {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].weights.len() - 1 >= weight_num);
    self.layer_storage[layer_index].weights[weight_num].clone()
  }

  pub fn get_biases(&self, layer_index: usize) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].biases.clone()
  }

  pub fn get_bias(&self, layer_index: usize, bias_num: usize) -> Array {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].biases.len() - 1 >= bias_num);
    self.layer_storage[layer_index].biases[bias_num].clone()
  }

  pub fn get_bias_dims(&self, layer_index: usize) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for b in &self.layer_storage[layer_index].biases {
      dims.push(b.dims().unwrap().clone());
    }
    dims
  }

  pub fn get_all_weight_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for layer in &self.layer_storage {
      for w in &layer.weights {
        dims.push(w.dims().unwrap().clone());
      }
    }
    dims
  }

  pub fn get_all_bias_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for layer in &self.layer_storage {
      for b in &layer.biases {
        dims.push(b.dims().unwrap().clone());
      }
    }
    dims
  }

  pub fn get_weight_dims(&self, layer_index: usize) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for w in &self.layer_storage[layer_index].weights {
      dims.push(w.dims().unwrap().clone());
    }
    dims
  }

  pub fn set_weights(&mut self, layer_index: usize, weights: Vec<Array>){
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].weights = weights;
  }

  pub fn set_weight(&mut self, layer_index: usize, weight_num: usize, weight: Array){
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].weights.len() - 1 >= weight_num);
    self.layer_storage[layer_index].weights[weight_num] = weight;
  }

  pub fn set_biases(&mut self, layer_index: usize, biases: Vec<Array>){
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].biases = biases;
  }

  pub fn set_bias(&mut self, layer_index: usize, bias_num: usize, bias: Array){
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].biases.len() - 1 >= bias_num);
    self.layer_storage[layer_index].biases[bias_num] = bias;
  }

  pub fn get_deltas(&self, layer_index: usize) -> Vec<Array> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].deltas.clone()
  }

  pub fn get_delta(&self, layer_index: usize, delta_num: usize) -> Array {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].deltas.len() - 1 >= delta_num);
    self.layer_storage[layer_index].deltas[delta_num].clone()
  }

  pub fn set_deltas(&mut self, layer_index: usize, deltas: Vec<Array>) {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].deltas = deltas;
  }

  pub fn set_delta(&mut self, layer_index: usize, delta_num: usize, delta: Array) {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].deltas.len() - 1 >= delta_num);
    self.layer_storage[layer_index].deltas[delta_num] = delta;
  }

  pub fn get_inputs(&self, layer_index: usize) -> Vec<Input> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].inputs.clone()
  }

  pub fn get_input(&self, layer_index: usize, input_num: usize) -> Input {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].inputs.len() - 1 >= input_num);
    self.layer_storage[layer_index].inputs[input_num].clone()
  }

  pub fn set_inputs(&mut self, layer_index: usize, input_num: usize, input: Vec<Input>) {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].inputs.len() - 1 >= input_num);
    self.layer_storage[layer_index].inputs = input;
  }

  pub fn set_input(&mut self, layer_index: usize, input_num: usize, input: Input) {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].inputs.len() - 1 >= input_num);
    self.layer_storage[layer_index].inputs[input_num] = input;
  }

  pub fn get_outputs(&self, layer_index: usize) -> Vec<Input> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].outputs.clone()
  }

  pub fn get_output(&self, layer_index: usize, output_num: usize) -> Input {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].outputs.len() - 1 >= output_num);
    self.layer_storage[layer_index].inputs[output_num].clone()
  }

  pub fn set_outputs(&mut self, layer_index: usize, output_num: usize, output: Vec<Input>) {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].outputs.len() - 1 >= output_num);
    self.layer_storage[layer_index].outputs = output;
  }

  pub fn set_output(&mut self, layer_index: usize, output_num: usize, output: Input) {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].outputs.len() - 1 >= output_num);
    self.layer_storage[layer_index].inputs[output_num] = output;
  }

  pub fn get_activations(&self, layer_index: usize) -> Vec<String> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.layer_storage[layer_index].activations.clone()
  }

  pub fn get_activation(&self, layer_index: usize, activation_num: usize) -> String {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    assert!(self.layer_storage[layer_index].activations.len() - 1 >= activation_num);
    self.layer_storage[layer_index].activations[activation_num].clone()
  }


  pub fn tie_weight(&mut self, layer_input: usize, iweight_index: usize
                , layer_output: usize, oweight_index: usize)
  {
    assert!(self.layer_storage.len() - 1 >= layer_input);
    assert!(self.layer_storage.len() - 1 >= layer_output);
    assert!(self.layer_storage[layer_input].weights.len() - 1 >= iweight_index);
    assert!(self.layer_storage[layer_output].weights.len() - 1 >= oweight_index);
    let input_dims = self.layer_storage[layer_input].weights[iweight_index].dims().unwrap();
    let output_dims = self.layer_storage[layer_output].weights[oweight_index].dims().unwrap();
    assert!((input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1])
            || (input_dims[0] == output_dims[1] && input_dims[1] == output_dims[0]));
    self.layer_storage[layer_output].weights[oweight_index]
      = self.layer_storage[layer_input].weights[iweight_index].clone();
  }

  pub fn tie_bias(&mut self, layer_input: usize, ibias_index: usize
              , layer_output: usize, obias_index: usize)
  {
    assert!(self.layer_storage.len() - 1 >= layer_input);
    assert!(self.layer_storage.len() - 1 >= layer_output);
    assert!(self.layer_storage[layer_input].biases.len() - 1 >= ibias_index);
    assert!(self.layer_storage[layer_output].biases.len() - 1 >= obias_index);
    let input_dims = self.layer_storage[layer_input].biases[ibias_index].dims().unwrap();
    let output_dims = self.layer_storage[layer_output].biases[obias_index].dims().unwrap();
    assert!(input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1]);
    self.layer_storage[layer_output].biases[obias_index]
      = self.layer_storage[layer_input].biases[ibias_index].clone();
  }
}

/** Custom Layer Traits **/
pub trait DenseGenerator {
  fn add_dense(&mut self
               , input_size: usize
               , output_size: usize
               , activation: &str
               , w_init: &str
               , b_init: &str);

}

pub enum LSTMIndex {
  Input,      // i_t
  Forget,     // f_t
  Output,     // o_t
  CellTilda,  // ct_t
  Cell,       // c_t
  CellOutput, // h_t
}

pub trait LSTMGenerator {
  fn add_lstm(&mut self
              , input_size: usize
              , output_size: usize
              , max_seq_size: usize
              , input_activation: &str
              , output_activation: &str
              , w_inner_init: &str
              , w_outer_init: &str
              , forget_bias_init: &str
              , b_init: &str);
  fn get_recurrences(&self, layer_index: usize) -> Vec<Array>;
  fn get_recurrence(&self, layer_index: usize, recur_name: LSTMIndex) -> Array;
  fn set_recurrences(&mut self, layer_index: usize, recurrences: Vec<Array>);
  fn set_recurrence(&mut self, layer_index: usize, recur_name: LSTMIndex, recurrence: Array);
}

/** Custom Layer Impls **/
impl DenseGenerator for ParamManager {
  fn add_dense(&mut self
               , input_size: usize
               , output_size: usize
               , activation: &str
               , w_init: &str
               , b_init: &str)
  {
    self.add("dense"
             , vec![w_init]
             , vec![(output_size, input_size)]
             , vec![b_init]
             , vec![(output_size, 1)]
             , vec![activation]
             , None, None);
  }
}

impl LSTMGenerator for ParamManager {
  fn add_lstm(&mut self
              , input_size: usize
              , output_size: usize
              , max_seq_size: usize
              , inner_activation: &str
              , outer_activation: &str
              , w_init: &str
              , w_recurrent_init: &str
              , forget_b_init: &str
              , b_init: &str)
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
             , vec![inner_activation, outer_activation]
             , Some(vec![bias_dims; 6])
             , Some(vec![("zeros", input_dims)       // dW
                         , ("zeros", recurrent_dims) // dU
                         , ("zeros", bias_dims)]));  // db
  }

  fn get_recurrences(&self, layer_index: usize) -> Vec<Array>{
    let offset = 4;
    vec![self.get_weight(layer_index, LSTMIndex::Input as usize + offset)
         , self.get_weight(layer_index, LSTMIndex::Forget as usize + offset)
         , self.get_weight(layer_index, LSTMIndex::Output as usize+ offset)
         , self.get_weight(layer_index, LSTMIndex::CellTilda as usize + offset)]
  }

  fn get_recurrence(&self, layer_index: usize, recur_name: LSTMIndex) -> Array{
    let offset = 4;
    self.get_weight(layer_index, recur_name as usize + offset)
  }

  fn set_recurrences(&mut self, layer_index: usize, recurrences: Vec<Array>){
    self.set_recurrence(layer_index, LSTMIndex::Input, recurrences[LSTMIndex::Input as usize].clone());
    self.set_recurrence(layer_index, LSTMIndex::Forget, recurrences[LSTMIndex::Forget as usize].clone());
    self.set_recurrence(layer_index, LSTMIndex::Output, recurrences[LSTMIndex::Output as usize].clone());
    self.set_recurrence(layer_index, LSTMIndex::CellTilda, recurrences[LSTMIndex::CellTilda as usize].clone());
  }

  fn set_recurrence(&mut self, layer_index: usize, recur_name: LSTMIndex, recurrence: Array){
    let offset = 4;
    self.set_weight(layer_index, offset + recur_name as usize, recurrence);
  }
}
