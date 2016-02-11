use af::{Array, Dim4};
use std::default::Default;
//use itertools::Zip;

use initializations;
use device::{Device, DeviceManager};
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
  pub device: Device,
  pub weights: Vec<Array>,
  pub biases: Vec<Array>,
  pub activations: Vec<String>,
  pub deltas: Vec<Array>,
  pub inputs: Vec<Input>,
  pub outputs: Vec<Input>,
  pub recurrences: Vec<Array>,
  pub optional: Vec<Array>,
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
             , manager: DeviceManager
             , device: Device
             , layer_type: &str
             , weight_params: Vec<(&str, (usize, usize))> //(init, (i, o))
             , biases_params: Vec<(&str, (usize, usize))> //(init, (i, o))
             //, delta_params:  Vec<(&str, (usize, usize))>    //(init, (i, o))
             , activations: Vec<&str>
             , recurrence_dims: Option<Vec<(&str, (usize, usize))>>
             , optional_dims: Option<Vec<(&str, (usize, usize))>>)
  {
    // toggle device to appropriate one
    manager.swap_device(device);

    // generate the weights
    let mut weights: Vec<Array> = Vec::with_capacity(weight_params.len());
    for (w_init, w_dims) in weight_params {
      weights.push(self.generate(w_init, w_dims));
    }
    // generate the biases
    let mut biases: Vec<Array> = Vec::with_capacity(biases_params.len());
    for (b_init, b_dims) in biases_params {
      //println!("orig bias size: {:?}", b_dims);
      biases.push(self.generate(b_init, b_dims));
    }

    // // generate the deltas
    // let mut deltas: Vec<Array> = Vec::with_capacity(delta_params.len());
    // for (d_init, d_dims) in delta_params {
    //   deltas.push(self.generate(d_init, d_dims));
    // }


    // generate recurrence vectors
    let mut recurrences: Vec<Array> = Vec::new();
    if let Some(r) = recurrence_dims{
      for (r_init, r_dims) in r {
        recurrences.push(self.generate(r_init, r_dims));
      }
    }

    // some elements have optional params
    let mut optional: Vec<Array> = Vec::new();
    if let Some(o) = optional_dims {
      for (o_init, o_dims) in o {
        optional.push(self.generate(o_init, o_dims));
      }
    }

    let owned_activations = activations.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    self.layer_storage.push(Params{
      layer_type: layer_type.to_string(),
      device: device,
      weights: weights,
      biases: biases,
      activations: owned_activations,
      deltas: Vec::new(),
      inputs: Vec::new(),
      outputs: Vec::new(),
      recurrences: recurrences,
      optional: optional,
    });
  }

  fn generate(&self, init: &str, dims: (usize, usize)) -> Array {
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

  get_param_func!(get_weight, weights, Array);
  get_param_func!(get_bias, biases, Array);
  get_param_func!(get_activation, activations, String);
  get_param_func!(get_delta, deltas, Array);
  get_param_func!(get_input, inputs, Input);
  get_param_func!(get_output, outputs, Input);
  get_param_func!(get_recurrence, recurrences, Array);
  get_param_func!(get_optional, optional, Array);

  get_mut_param_func!(get_mut_weight, weights, Array);
  get_mut_param_func!(get_mut_bias, biases, Array);
  get_mut_param_func!(get_mut_activation, activations, String);
  get_mut_param_func!(get_mut_delta, deltas, Array);
  get_mut_param_func!(get_mut_input, inputs, Input);
  get_mut_param_func!(get_mut_output, outputs, Input);
  get_mut_param_func!(get_mut_recurrence, recurrences, Array);
  get_mut_param_func!(get_mut_optional, optional, Array);

  get_param_vec_func!(get_weights, weights, Array);
  get_param_vec_func!(get_biases, biases, Array);
  get_param_vec_func!(get_activations, activations, String);
  get_param_vec_func!(get_deltas, deltas, Array);
  get_param_vec_func!(get_inputs, inputs, Input);
  get_param_vec_func!(get_outputs, outputs, Input);
  get_param_vec_func!(get_recurrences, recurrences, Array);
  get_param_vec_func!(get_optionals, optional, Array);

  get_param_vec_func!(get_mut_weights, weights, Array);
  get_param_vec_func!(get_mut_biases, biases, Array);
  get_param_vec_func!(get_mut_activations, activations, String);
  get_param_vec_func!(get_mut_deltas, deltas, Array);
  get_param_vec_func!(get_mut_inputs, inputs, Input);
  get_param_vec_func!(get_mut_outputs, outputs, Input);
  get_param_vec_func!(get_mut_recurrences, recurrences, Array);
  get_param_vec_func!(get_mut_optionals, optional, Array);

  set_param_func!(set_weight, weights, Array);
  set_param_func!(set_bias, biases, Array);
  set_param_func!(set_activation, activations, String);
  set_param_func!(set_delta, deltas, Array);
  set_param_func!(set_input, inputs, Input);
  set_param_func!(set_output, outputs, Input);
  set_param_func!(set_recurrence, recurrences, Array);
  set_param_func!(set_optional, optional, Array);

  set_param_vec_func!(set_weights, weights, Array);
  set_param_vec_func!(set_biases, biases, Array);
  set_param_vec_func!(set_activations, activations, String);
  set_param_vec_func!(set_deltas, deltas, Array);
  set_param_vec_func!(set_inputs, inputs, Input);
  set_param_vec_func!(set_outputs, outputs, Input);
  set_param_vec_func!(set_recurrences, recurrences, Array);
  set_param_vec_func!(set_optionals, optional, Array);

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
               , manager: DeviceManager
               , device: Device
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
              , manager: DeviceManager
              , device: Device
              , input_size: usize
              , output_size: usize
              , max_seq_size: usize
              , input_activation: &str
              , output_activation: &str
              , w_inner_init: &str
              , w_outer_init: &str
              , forget_bias_init: &str
              , b_init: &str);
}

/** Custom Layer Impls **/
impl<'a> DenseGenerator for ParamManager {
  fn add_dense(&mut self
               , manager: DeviceManager
               , device: Device
               , input_size: usize
               , output_size: usize
               , activation: &str
               , w_init: &str
               , b_init: &str)
  {
    self.add(manager, device, "dense"
             , vec![(w_init, (input_size, output_size))]
             , vec![(b_init, (output_size, 1))]
             , vec![activation]
             , None, None);
  }
}

impl LSTMGenerator for ParamManager {
  fn add_lstm(&mut self
              , manager: DeviceManager
              , device: Device
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
    let input_dims = (input_size, output_size);
    let recurrent_dims = (output_size, output_size);
    let bias_dims = (output_size, 1);
    // W_i, W_f, W_o, W_ct, U_i, U_f, U_o, U_ct
    self.add(manager, device, "lstm"
             , vec![(w_init, input_dims)
                    , (w_init, input_dims)
                    , (w_init, input_dims)
                    , (w_init, input_dims)
                    , (w_recurrent_init, recurrent_dims)
                    , (w_recurrent_init, recurrent_dims)
                    , (w_recurrent_init, recurrent_dims)
                    , (w_recurrent_init, recurrent_dims)]
             , vec![(b_init, bias_dims)
                    , (forget_b_init, bias_dims)
                    , (b_init, bias_dims)
                    , (b_init, bias_dims)]
             , vec![inner_activation, outer_activation]
             , Some(vec![("zeros", bias_dims); 6])   // i_f_o_ct_c_h @ t-1
             , Some(vec![("zeros", input_dims)       // dW
                         , ("zeros", recurrent_dims) // dU
                         , ("zeros", bias_dims)]));  // db
  }
}
