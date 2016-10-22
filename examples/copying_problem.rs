#[macro_use] extern crate hal;
extern crate arrayfire as af;
extern crate rand;
use rand::distributions::{Range, IndependentSample};

use hal::utils;
use hal::Model;
use hal::loss;
use hal::initializations::uniform;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, AddingProblemSource, CopyingProblemSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Backend, HasAfEnum, MatProp, DType, Dim4};

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::io::prelude::*;

fn main() {
  // First we need to parameterize our network
  let input_dims = 10;
  let seq_size = 10;
  let hidden_dims = 30;
  let output_dims = 10;
  let num_train_samples = 100;
  let batch_size = 5;
  let optimizer_type = "Adam";
  let epochs = 1;
  let bptt_unroll = 120;
  let data_type = DType::F32;

  let num_test_samples = 1000;

  // Now, let's build a model with an device manager on a specific device
  // an optimizer and a loss function. For this example we demonstrate a simple autoencoder
  // AF_BACKEND_DEFAULT is: OpenCL -> CUDA -> CPU
  let manager = DeviceManagerFactory::new();
  let gpu_device = Device{backend: Backend::DEFAULT, id: 0};
  let cpu_device = Device{backend: Backend::CPU, id: 0};
  let optimizer = get_optimizer_with_defaults(optimizer_type).unwrap();
  let loss_fct = "cross_entropy_softmax";
  let mut model = Box::new(Sequential::new(manager.clone()
                                           , optimizer         // optimizer
                                           , loss_fct          // loss
                                           , gpu_device));     // device for model

  // Add the unitary layer
  model.add::<f32>("unitary", hashmap!["input_size"   => input_dims.to_string()
                   , "output_size"       => output_dims.to_string()
                   , "hidden_size"       => hidden_dims.to_string()
                   , "o_activation"      => "ones".to_string()
                   , "h_init"            => "glorot_uniform".to_string()
                   , "v_init"            => "glorot_uniform".to_string()
                   , "phase_init"        => "glorot_uniform".to_string()
                   , "householder_init"  => "glorot_uniform".to_string()
                   , "u_init"            => "glorot_uniform".to_string()
                   , "h_bias_init"       => "zeros".to_string()
                   , "o_bias_init"       => "zeros".to_string()
                   , "is_permut_const"   => "false".to_string()]);


  model.info();
  manager.swap_device(cpu_device);

  // Initializes the generator of copying problem data
  let train_generator = CopyingProblemSource::new(input_dims
                                                  , batch_size
                                                  , seq_size
                                                  , bptt_unroll
                                                  , data_type
                                                  , num_train_samples);

  // Training process
  let loss = model.fit::<CopyingProblemSource, f32>(&train_generator
                                                    , cpu_device
                                                    , epochs
                                                    , batch_size
                                                    , Some(bptt_unroll)
                                                    , None
                                                    , true);

}
