#[macro_use] extern crate hal;
extern crate arrayfire as af;

use hal::Model;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, MultUniformSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Backend};


fn main() {
  // First we need to parameterize our network
  let input_dims = 2;
  let hidden_dims = 3;
  let output_dims = 2;
  let num_train_samples = 5;
  let batch_size = 3;
  let optimizer_type = "SGD";
  let epochs = 5;
  let bptt_unroll = 10;

  // Now, let's build a model with an device manager on a specific device
  // an optimizer and a loss function. For this example we demonstrate a simple autoencoder
  // AF_BACKEND_DEFAULT is: OpenCL -> CUDA -> CPU
  let manager = DeviceManagerFactory::new();
  let gpu_device = Device{backend: Backend::DEFAULT, id: 0};
  let cpu_device = Device{backend: Backend::CPU, id: 0};
  let optimizer = get_optimizer_with_defaults(optimizer_type).unwrap();
  let mut model = Box::new(Sequential::new(manager.clone()
                                           , optimizer         // optimizer
                                           , "mse"             // loss
                                           , gpu_device));     // device for model

  // Let's add a few layers why don't we?
  model.add::<f32>("unitary", hashmap!["input_size"   => input_dims.to_string()
                                     , "output_size"  => output_dims.to_string()
                                     , "hidden_size"  => hidden_dims.to_string()
                                     , "h_activation" => "relu".to_string()
                                     , "o_activation" => "tanh".to_string()
                                     , "h_init"       => "glorot_uniform".to_string()
                                     , "v_init"       => "glorot_uniform".to_string()
                                     , "phase_init"      => "glorot_uniform".to_string()
                                     , "permut_init"      => "permut".to_string()
                                     , "householder_init"      => "glorot_uniform".to_string()
                                     , "u_init"       => "glorot_uniform".to_string()
                                     , "h_bias_init"      => "zeros".to_string()
                                     , "o_bias_init"      => "zeros".to_string()]);


  manager.swap_device(cpu_device);

  // Build our sin wave source
  let uniform_generator = MultUniformSource::new(input_dims, output_dims
                                                 , batch_size
                                                 , bptt_unroll
                                                 , num_train_samples
                                                 , false   // normalized
                                                 , false); // shuffled

  // Pull a sample to verify sizing
  let minibatch = uniform_generator.get_train_iter(batch_size);
  let batch_input = minibatch.input.into_inner();
    


  model.forward::<f32>(&batch_input, cpu_device, cpu_device, true);

  let params_arc = model.param_manager.get_params(0);

  let params = params_arc.lock().unwrap();

  //af::print(&batch_input);
  
  
  /*
  
  for i in 0..params.weights.len() {
       println!("{}", i);
      af::print(&params.weights[i]);
  }
  */
  
  /*
  for i in 0..params.biases.len() {
      println!("{}", i);
      af::print(&params.biases[i]);
  }
  */

  
  for i in 0..params.recurrences.len() {
      println!("{}", i);
      af::print(&params.recurrences[i]);
  }
  

  /*
  for i in 0..params.outputs.len() {
      println!("{}", i);
      af::print(&params.outputs[i].data);
  }
  */


  
  //af::print(&params.optional[0]);



}




