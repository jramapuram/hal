use af::{Array, Dim4, HasAfEnum, DType};
use std::default::Default;
use std::sync::{Arc, Mutex};

use utils;
use initializations;
use device::{Device, DeviceManager};
//use error::HAL Error;

macro_rules! set_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&self, layer_index: usize, p: Vec<$base_type>) {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      let layer = self.layer_storage[layer_index].clone();
      let mut ltex = &mut layer.lock().unwrap();
      ltex.$vec_extension = p;
    }
    )
}

macro_rules! get_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&self, layer_index: usize) -> Vec<$base_type> {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      let layer = self.layer_storage[layer_index].clone();
      let ltex = layer.lock().unwrap();
      ltex.$vec_extension.clone()
    }
    )
}

macro_rules! with_mut_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name<F>(&mut self, layer_index: usize, mut f: F)
      where F: FnMut(&mut Vec<$base_type>)
    {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      let layer = self.layer_storage[layer_index].clone();
      f(&mut layer.lock().unwrap().$vec_extension);
    }
    )
}

macro_rules! set_param_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&self, layer_index: usize, num: usize, p: $base_type) {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      let layer = self.layer_storage[layer_index].clone();
      let mut ltex = layer.lock().unwrap();
      let mut ext = &mut ltex.$vec_extension;
      assert!(ext.len() - 1 >= num);
      ext[num] = p;
    }
    )
}

macro_rules! get_param_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&self, layer_index: usize, num: usize) -> $base_type {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      let layer = self.layer_storage[layer_index].clone();
      let mut ltex = layer.lock().unwrap();
      let mut ext = &ltex.$vec_extension;
      assert!(ext.len() - 1 >= num);
      ext[num].clone()
    }
    )
}

#[derive(Clone)]
pub struct Params {
  pub layer_type: String,
  pub device: Device,
  pub weights: Vec<Array>,
  pub biases: Vec<Array>,
  pub activations: Vec<String>,
  pub deltas: Vec<Array>,
  pub inputs: Vec<Array>,
  pub outputs: Vec<Array>,
  pub recurrences: Vec<Array>,
  pub state_derivatives: Vec<Array>,
  pub current_unroll: usize,
  pub optional: Vec<Array>,
}

pub struct ParamManager {
  layer_storage: Vec<Arc<Mutex<Params>>>,
}

impl Default for ParamManager {
  fn default() -> ParamManager {
    ParamManager {
      layer_storage: Vec::new(),
    }
  }
}

impl ParamManager {
  pub fn add<T: HasAfEnum>(&mut self
                           , manager: DeviceManager
                           , device: Device
                           , layer_type: &str
                           , weight_params: Vec<(&str, (usize, usize))> //(init, (i, o))
                           , biases_params: Vec<(&str, (usize, usize))> //(init, (i, o))
                           , activations: Vec<&str>
                           , recurrence_dims: Option<Vec<(&str, (usize, usize))>>
                           , optional_dims: Option<Vec<(&str, (usize, usize))>>)
  {
    // toggle device to appropriate one
    manager.swap_device(device);
    let num_params = weight_params.len() + biases_params.len();

    // allocate deltas here so that they can be pushed in at each W/b add
    let mut deltas: Vec<Array> = Vec::with_capacity(num_params);

    // generate the weights
    let mut weights: Vec<Array> = Vec::with_capacity(weight_params.len());
    for (w_init, w_dims) in weight_params {
      weights.push(self.generate::<T>(w_init, w_dims));
      deltas.push(self.generate::<T>("zeros", w_dims));
    }
    // generate the biases
    let mut biases: Vec<Array> = Vec::with_capacity(biases_params.len());
    for (b_init, b_dims) in biases_params {
      //println!("orig bias size: {:?}", b_dims);
      biases.push(self.generate::<T>(b_init, b_dims));
      deltas.push(self.generate::<T>("zeros", b_dims));
    }

    // generate recurrence vectors
    // if the length of the recurrences are > 0 then init the inp/outputs
    let mut recurrences: Vec<Array> = Vec::new();
    if let Some(r) = recurrence_dims{
      for (r_init, r_dims) in r {
        recurrences.push(self.generate::<T>(r_init, r_dims));
      }
    }

    // some elements have optional params
    let mut optional: Vec<Array> = Vec::new();
    if let Some(o) = optional_dims {
      for (o_init, o_dims) in o {
        optional.push(self.generate::<T>(o_init, o_dims));
      }
    }

    let owned_activations = activations.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    self.layer_storage.push(Arc::new(Mutex::new(Params{
      layer_type: layer_type.to_string(),
      device: device,
      weights: weights,
      biases: biases,
      activations: owned_activations,
      deltas: deltas,
      inputs: Vec::new(),
      outputs: Vec::new(),
      recurrences: recurrences,
      state_derivatives: Vec::new(),
      current_unroll: 0,
      optional: optional,
    })));
  }

  fn generate<T: HasAfEnum>(&self, init: &str, dims: (usize, usize)) -> Array {
    let dims = Dim4::new(&[dims.0 as u64, dims.1 as u64, 1, 1]);
    initializations::get_initialization::<T>(init, dims).unwrap()
  }

  pub fn num_layers(&self) -> usize {
    self.layer_storage.len()
  }

  pub fn num_weights(&self, layer_index: usize) -> usize {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let layer = self.layer_storage[layer_index].clone();
    let ltex = layer.lock().unwrap();
    ltex.weights.len()
  }

  pub fn num_biases(&self, layer_index: usize) -> usize {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let layer = self.layer_storage[layer_index].clone();
    let ltex = layer.lock().unwrap();
    ltex.biases.len()
  }

  pub fn num_arrays(&self, layer_index: usize) -> usize {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    self.num_biases(layer_index) + self.num_weights(layer_index)
  }

  pub fn num_recurrences(&self, layer_index: usize) -> usize {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let layer = self.layer_storage[layer_index].clone();
    let ltex = layer.lock().unwrap();
    ltex.recurrences.len()
  }

  pub fn num_state_derivatives(&self, layer_index: usize) -> usize {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let layer = self.layer_storage[layer_index].clone();
    let ltex = layer.lock().unwrap();
    ltex.state_derivatives.len()
  }


  pub fn get_params(&self, layer_index: usize) -> Arc<Mutex<Params>> {
    assert!(self.layer_storage.len() - 1>= layer_index);
    self.layer_storage[layer_index].clone()
  }

  pub fn with_mut_params<F>(&self, layer_index: usize, mut f: F)
    where F: FnMut(&Params)
  {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let layer = self.layer_storage[layer_index].clone();
    f(&mut layer.lock().unwrap());
  }

  pub fn get_all_arrays(&self) -> Vec<Array> {
    let mut p = Vec::new();
    for layer_num in 0..self.num_layers() {
      p.extend(self.get_weights(layer_num));
      p.extend(self.get_biases(layer_num));
    }
    p
  }

  // assumes params are coming in layer wise
  // eg: [W0, b0, .. , WN, bN]
  pub fn set_array_from_index(&self, arr: Array, ind: usize) {
    let mut current: usize = 0;
    for layer_num in 0..self.num_layers() {
      let n_weights = self.num_weights(layer_num);
      let n_biases = self.num_biases(layer_num);

      if current + n_weights > ind { // we are a weight
        let w_index = ind - current;
        let target_dims = self.get_weight(layer_num, w_index).dims();
        let src_dims = arr.dims();
        assert!(src_dims == target_dims
                , "array at index {} does not match provided [provided: {:?}  internal: {:?}]"
                , ind, src_dims, target_dims);
        self.set_weight(layer_num, w_index, arr);
        break;
      }

      current += n_weights;
      if current + n_biases > ind { // we are a bias
        let b_index = ind - current;
        assert!(self.get_bias(layer_num, b_index).dims()
                == arr.dims());
        self.set_bias(layer_num, b_index, arr);
        break;
      }
      current += n_biases;
    }
  }

  // TODO:
  // pub fn get_mut_all_arrays(&mut self) -> Vec<&mut Array> {
  //   let mut p = Vec::new();
  //   for layer_num in 0..self.num_layers() {
  //     // p.extend(self.get_mut_weights(layer_num));
  //     // p.extend(self.get_mut_biases(layer_num));
  //     let mut storage = self.layer_storage[layer_num];
  //     p.push_all(&mut storage.weights[..]);
  //     p.push_all(&mut storage.biases[..]);
  //   }
  //   p
  // }


  // assumes params are coming in layer wise
  // eg: [W0, b0, .. , WN, bN]
  pub fn set_all_arrays(&mut self, params: Vec<Array>) {
    let mut index: usize = 0;
    for layer_num in 0..self.num_layers() {
      let n_weights = self.num_weights(layer_num);
      let n_biases = self.num_biases(layer_num);
      self.set_weights(layer_num, params[index..index+n_weights].to_vec());
      index += n_weights;
      self.set_biases(layer_num, params[index..index+n_biases].to_vec());
      index += n_biases;
    }
  }

  pub fn get_all_deltas(&self) -> Vec<Array> {
    let mut d = Vec::new();
    for layer_num in 0..self.num_layers() {
      d.extend(self.get_deltas(layer_num));
    }
    d
  }

  pub fn zero_all_deltas(&self) {
    for layer_num in 0..self.num_layers() {
      for delta_num in 0..self.num_arrays(layer_num) {
        let delta = self.get_delta(layer_num, delta_num);
        let delta_dims = delta.dims();
        let dtype = delta.get_type();
        let zero_tensor = utils::constant(delta_dims, dtype, 0.0f32);
        self.set_delta(layer_num, delta_num, zero_tensor);
      }
    }
  }

  pub fn zero_all_state_derivatives(&self) {
    for layer_num in 0..self.num_layers() {
      for state_num in 0..self.num_state_derivatives(layer_num) {
        let state_derivative = self.get_state_derivative(layer_num, state_num);
        let state_dims = state_derivative.dims();
        let dtype = state_derivative.get_type();
        let zero_tensor = utils::constant(state_dims, dtype, 0.0f32);
        self.set_state_derivative(layer_num, state_num, zero_tensor);
      }
    }
  }


  pub fn zero_all_states(&self, default_state: Option<Array>)
  {
    for layer_num in 0..self.num_layers() {
      for recurrence_num in 0..self.num_recurrences(layer_num) {
        match default_state {
          Some(ref st)  => self.set_recurrence(layer_num, recurrence_num, st.copy()),
          None          => {
            let recurrence = self.get_recurrence(layer_num, recurrence_num);
            let recurrence_dims = recurrence.dims();
            let dtype = recurrence.get_type();
            let zero_tensor = utils::constant(recurrence_dims, dtype, 0.0f32);
            self.set_recurrence(layer_num, recurrence_num, zero_tensor);
          },
        };
      }
    }
  }

  get_param_func!(get_weight, weights, Array);
  get_param_func!(get_bias, biases, Array);
  get_param_func!(get_activation, activations, String);
  get_param_func!(get_delta, deltas, Array);
  get_param_func!(get_input, inputs, Array);
  get_param_func!(get_output, outputs, Array);
  get_param_func!(get_recurrence, recurrences, Array);
  get_param_func!(get_state_derivative, state_derivatives, Array);
  get_param_func!(get_optional, optional, Array);

  get_param_vec_func!(get_weights, weights, Array);
  get_param_vec_func!(get_biases, biases, Array);
  get_param_vec_func!(get_activations, activations, String);
  get_param_vec_func!(get_deltas, deltas, Array);
  get_param_vec_func!(get_inputs, inputs, Array);
  get_param_vec_func!(get_outputs, outputs, Array);
  get_param_vec_func!(get_recurrences, recurrences, Array);
  get_param_vec_func!(get_state_derivatives, state_derivatives, Array);
  get_param_vec_func!(get_optionals, optional, Array);

  with_mut_param_vec_func!(with_mut_weights, weights, Array);
  with_mut_param_vec_func!(with_mut_biases, biases, Array);
  with_mut_param_vec_func!(with_mut_activations, activations, String);
  with_mut_param_vec_func!(with_mut_deltas, deltas, Array);
  with_mut_param_vec_func!(with_mut_inputs, inputs, Array);
  with_mut_param_vec_func!(with_mut_outputs, outputs, Array);
  with_mut_param_vec_func!(with_mut_recurrences, recurrences, Array);
  with_mut_param_vec_func!(with_mut_state_derivatives, state_derivatives, Array);
  with_mut_param_vec_func!(with_mut_optionals, optional, Array);

  set_param_func!(set_weight, weights, Array);
  set_param_func!(set_bias, biases, Array);
  set_param_func!(set_activation, activations, String);
  set_param_func!(set_delta, deltas, Array);
  set_param_func!(set_input, inputs, Array);
  set_param_func!(set_output, outputs, Array);
  set_param_func!(set_recurrence, recurrences, Array);
  set_param_func!(set_state_derivative, state_derivatives, Array);
  set_param_func!(set_optional, optional, Array);

  set_param_vec_func!(set_weights, weights, Array);
  set_param_vec_func!(set_biases, biases, Array);
  set_param_vec_func!(set_activations, activations, String);
  set_param_vec_func!(set_deltas, deltas, Array);
  set_param_vec_func!(set_inputs, inputs, Array);
  set_param_vec_func!(set_outputs, outputs, Array);
  set_param_vec_func!(set_recurrences, recurrences, Array);
  set_param_vec_func!(set_state_derivatives, state_derivatives, Array);
  set_param_vec_func!(set_optionals, optional, Array);

  pub fn get_bias_dims(&self, layer_index: usize) -> Vec<Dim4> {
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let mut dims = Vec::new();
    let layer = self.layer_storage[layer_index].clone();
    for b in &layer.lock().unwrap().biases {
      dims.push(b.dims().clone());
    }
    dims
  }

  pub fn get_all_weight_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for layer in &self.layer_storage {
      let ltex = layer.lock().unwrap();
      for w in &ltex.weights {
        dims.push(w.dims().clone());
      }
    }
    dims
  }

  pub fn get_all_bias_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for layer in &self.layer_storage {
      let ltex = layer.lock().unwrap();
      for b in &ltex.biases {
        dims.push(b.dims().clone());
      }
    }
    dims
  }

  pub fn get_all_dims(&self) -> Vec<Dim4> {
    let mut dims = Vec::new();
    for layer in &self.layer_storage {
      let ltex = layer.lock().unwrap();
      for w in &ltex.weights {
        dims.push(w.dims().clone());
      }
      for b in &ltex.biases {
        dims.push(b.dims().clone());
      }
    }
    dims
  }


  pub fn get_weight_dims(&self, layer_index: usize) -> Vec<Dim4> {
    let mut dims = Vec::new();
    assert!(self.layer_storage.len() - 1 >= layer_index);
    let layer = self.layer_storage[layer_index].clone();
    for w in &layer.lock().unwrap().weights {
      dims.push(w.dims().clone());
    }
    dims
  }

  pub fn tie_weights(&mut self, layer_input: usize, iweight_index: usize
                     , layer_output: usize, oweight_index: usize)
  {
    assert!(self.layer_storage.len() - 1 >= layer_input);
    assert!(self.layer_storage.len() - 1 >= layer_output);

    let layer_src = self.layer_storage[layer_input].clone();
    let layer_dest = self.layer_storage[layer_output].clone();

    {
      let weights_src_len = layer_src.lock().unwrap().weights.len();
      let weights_dest_len = layer_dest.lock().unwrap().weights.len();
      assert!(weights_src_len - 1 >= iweight_index);
      assert!(weights_dest_len - 1 >= oweight_index);
    }


    {
      let iweights = layer_src.lock().unwrap();
      let oweights = layer_dest.lock().unwrap();
      let input_dims = iweights.weights[iweight_index].dims();
      let output_dims = oweights.weights[oweight_index].dims();
      assert!((input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1])
              || (input_dims[0] == output_dims[1] && input_dims[1] == output_dims[0]));
    }

    layer_dest.lock().unwrap().weights[oweight_index]
      = layer_src.lock().unwrap().weights[iweight_index].clone();
  }

  pub fn tie_bias(&mut self, layer_input: usize, ibias_index: usize
                  , layer_output: usize, obias_index: usize)
  {
    assert!(self.layer_storage.len() - 1 >= layer_input);
    assert!(self.layer_storage.len() - 1 >= layer_output);
    let layer_src = self.layer_storage[layer_input].clone();
    let layer_dest = self.layer_storage[layer_output].clone();

    {
      let biases_src_len = layer_src.lock().unwrap().biases.len();
      let biases_dest_len = layer_dest.lock().unwrap().biases.len();
      assert!(biases_src_len - 1 >= ibias_index);
      assert!(biases_dest_len - 1 >= obias_index);
    }

    {
      let input_dims = layer_src.lock().unwrap().biases[ibias_index].dims();
      let output_dims = layer_dest.lock().unwrap().biases[obias_index].dims();
      assert!(input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1]);
    }

    layer_dest.lock().unwrap().biases[obias_index]
      = layer_src.lock().unwrap().biases[ibias_index].clone();
  }
}

/** Custom Layer Traits **/
pub trait DenseGenerator {
  fn add_dense<T: HasAfEnum>(&mut self
                             , manager: DeviceManager
                             , device: Device
                             , input_size: usize
                             , output_size: usize
                             , activation: &str
                             , w_init: &str
                             , b_init: &str);

}

pub trait RNNGenerator {
  fn add_rnn<T: HasAfEnum>(&mut self
                           , manager: DeviceManager
                           , device: Device
                           , input_size: usize
                           , hidden_size: usize
                           , output_size: usize
                           , inner_activation: &str
                           , outer_activation: &str
                           , w_init: &str
                           , b_init: &str);
}

pub enum LSTMIndex {
  Input=0,    // i_t
  Forget,     // f_t
  Output,     // o_t
  CellTilda,  // ct_t
  Cell,       // c_t
  CellOutput, // h_t
}

pub enum RNNIndex {
  InputToHidden=0,
  HiddenToOutput,
  HiddenToHidden,
}

pub trait LSTMGenerator {
  fn add_lstm<T: HasAfEnum>(&mut self
                            , manager: DeviceManager
                            , device: Device
                            , input_size: usize
                            , output_size: usize
                            //, bptt_interval: usize
                            , input_activation: &str
                            , output_activation: &str
                            , w_init: &str
                            , w_recurrent_init: &str
                            , forget_bias_init: &str
                            , b_init: &str);
}

/** Custom Layer Impls **/
impl DenseGenerator for ParamManager {
  fn add_dense<T: HasAfEnum>(&mut self
                             , manager: DeviceManager
                             , device: Device
                             , input_size: usize
                             , output_size: usize
                             , activation: &str
                             , w_init: &str
                             , b_init: &str)
  {
    self.add::<T>(manager, device, "dense"
                  , vec![(w_init, (input_size, output_size))]
                  , vec![(b_init, (output_size, 1))]
                  , vec![activation]
                  , None, None);
  }
}

impl RNNGenerator for ParamManager {
  fn add_rnn<T: HasAfEnum>(&mut self
                           , manager: DeviceManager
                           , device: Device
                           , input_size: usize
                           , hidden_size: usize
                           , output_size: usize
                           , inner_activation: &str
                           , outer_activation: &str
                           , w_init: &str
                           , b_init: &str)
  {
    let recurrent_dims = (hidden_size, hidden_size);
    let input_dims = (input_size, hidden_size);
    let output_dims = (hidden_size, output_size);
    let input_bias_dims = (hidden_size, 1);
    let output_bias_dims = (output_size, 1);

    // all weights are passed as one to the add func
    let weights = vec![(w_init, input_dims)         // input 2 hidden
                       , (w_init, output_dims)      // hidden to output
                       , (w_init, recurrent_dims)]; // hidden to hidden
    let biases = vec![(b_init, input_bias_dims), (b_init, output_bias_dims)];

    self.add::<T>(manager, device, "rnn"
                  , weights                                            // weight dims
                  , biases                                             // bias dims
                  , vec![inner_activation, outer_activation]           // activation vector
                  , None
                  , None);
  }
}

impl LSTMGenerator for ParamManager {
  fn add_lstm<T: HasAfEnum>(&mut self
              , manager: DeviceManager
              , device: Device
              , input_size: usize
              , output_size: usize
              //, bptt_interval: usize
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
    self.add::<T>(manager, device, "lstm"
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
