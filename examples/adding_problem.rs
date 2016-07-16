#[macro_use] extern crate hal;
extern crate arrayfire as af;
extern crate rand;
use rand::distributions::{Range, IndependentSample};

use hal::utils;
use hal::Model;
use hal::initializations::uniform;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, AddingProblemSource, CopyingProblemSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Backend, HasAfEnum, MatProp, DType};


fn main() {
  // First we need to parameterize our network
  let input_dims = 1;
  let seq_size = 4;
  let hidden_dims = 30;
  let output_dims = 1;
  let num_train_samples = 32000;
  let batch_size = 3;
  let optimizer_type = "Adam";
  let epochs = 5;
  let bptt_unroll = 8;

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
                                     , "o_activation" => "ones".to_string()
                                     , "h_init"       => "glorot_uniform".to_string()
                                     , "v_init"       => "glorot_uniform".to_string()
                                     , "phase_init"      => "glorot_uniform".to_string()
                                     , "permut_init"      => "permut".to_string()
                                     , "householder_init"      => "glorot_uniform".to_string()
                                     , "u_init"       => "glorot_uniform".to_string()
                                     , "h_bias_init"      => "zeros".to_string()
                                     , "o_bias_init"      => "zeros".to_string()]);


  model.info();
  manager.swap_device(cpu_device);

  let uniform_generator = AddingProblemSource::new(batch_size
                                                 , bptt_unroll
                                                 , DType::F32
                                                 , num_train_samples);

  /*
  let loss = model.fit::<AddingProblemSource, f32>(&uniform_generator
                                                    , cpu_device
                                                    , epochs
                                                    , batch_size
                                                    , Some(bptt_unroll)
                                                    , true);
  */

  let minibatch = uniform_generator.get_train_iter(batch_size);
  let batch_input = minibatch.input.into_inner();
  let batch_target = minibatch.target.into_inner();
  af::print(&batch_input);
  af::print(&batch_target);
  
  
  /*
  let params_arc = model.param_manager.get_params(0);

  let params = params_arc.lock().unwrap();

  
  
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

  
  /*
  for i in 0..params.recurrences.len() {
      println!("{}", i);
      af::print(&params.recurrences[i]);
  }
  */
  

  /*
  for i in 0..params.outputs.len() {
      println!("{}", i);
      af::print(&params.outputs[i].data);
  }
  */


  /*
  let between = Range::new(0, 10);
  let mut rng = rand::thread_rng();

  for _ in 1..100{
  let a = between.ind_sample(&mut rng);
  println!("{}", a);
  }
 */

  /*
  let dim = af::Dim4::new(&[2,1,3,1]);
  let mut a = af::Array::new(&[0.,2.,3.,0.,5.,6.], dim);
  let mut b = af::constant(0., dim);

  af::print(&a);
  af::print(&b);

  let c = af::gt(&a, &b, false);

  af::print(&c);
  let d = af::select(&a, &c, &b);
  af::print(&d);
  */
*/




}




