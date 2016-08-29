#[macro_use] extern crate hal;
extern crate arrayfire as af;

use hal::Model;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, DataLoader, SinSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Backend, DType};
use std::sync::{Arc, Mutex};

fn main() {
  // First we need to parameterize our network
  let input_dims = 128;
  let hidden_dims = 64;
  let output_dims = 128;
  let num_train_samples = 65536;
  let batch_size = 128;
  let optimizer_type = "Adam";
  let epochs = 5;

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
  model.add::<f32>("dense", hashmap!["activation"    => "tanh".to_string()
                                     , "input_size"  => input_dims.to_string()
                                     , "output_size" => hidden_dims.to_string()
                                     , "w_init"      => "glorot_uniform".to_string()
                                     , "b_init"      => "zeros".to_string()]);
  model.add::<f32>("dense", hashmap!["activation"    => "linear".to_string()
                                     , "input_size"  => hidden_dims.to_string()
                                     , "output_size" => output_dims.to_string()
                                     , "w_init"      => "glorot_uniform".to_string()
                                     , "b_init"      => "zeros".to_string()]);

  // Get some nice information about our model
  model.info();

  // Temporarily set the backend to CPU so that we can load data into RAM
  // The model will automatically toggle to the desired backend during training
  manager.swap_device(cpu_device);

  // Bind a simple sin wave generating source
  let source = SinSource::new(input_dims, batch_size
                              , DType::F32         // specify the type of data
                              , num_train_samples  // generator can do inf, so decide
                              , false              // normalized ?
                              , false);

  // Note: Currently the DataLoader only works with the same source and target device
  // Build our sin wave source & wrap in a dataloader (thread based async queuing)
  // The dataloader only makes sense for data that is not sequence dependent
  // let source = Arc::new(Mutex::new(SinSource::new(input_dims, batch_size
  //                                                 , DType::F32         // specify the type of data
  //                                                 , num_train_samples  // generator can do inf, so decide
  //                                                 , false              // normalized ?
  //                                                 , false)));          // shuffled ?
  // let source = DataLoader::new(4                   // number of worker threads
  //                              , manager           // the device manager
  //                              , cpu_device        // data source device
  //                              , batch_size * 10   // training buffer size
  //                              , batch_size * 2    // test buffer size
  //                              , batch_size * 2    // validation buffer size
  //                              , batch_size        // self explanatory
  //                              , source.clone());  // the source to wrap


  // iterate our model in Verbose mode (printing loss)
  // Note: more manual control can be enacted by directly calling
  //       forward/backward & optimizer update [see sequential.rs]
  let loss = model.fit::<SinSource, f32>(&source              // what data source to pull from
                                         , cpu_device          // source device
                                         , epochs, batch_size  // self explanatory :)
                                         , None                // BPTT interval [rnn only]
                                         , None                // Custom loss indices
                                         , true);              // verbose

  // plot our loss on a 512x512 grid with the provided title
  plot_vec(loss, "Loss vs. Iterations", 512, 512);

  // infer on one test and plot the first sample (row) of the predictions
  let test_sample = source.get_test_iter(1).input.into_inner();
  println!("test sample shape: {:?}", test_sample.dims());
  let prediction = model.forward::<f32>(&test_sample
                                        , cpu_device   // source device
                                        , cpu_device); // destination device
  println!("\nprediction shape: {:?} | output backend = {:?}"
           , prediction[0].dims(), prediction[0].get_backend());

  // plot our inference
  plot_array(&af::flat(&prediction[0]), "Model Inference", 512, 512);
}
