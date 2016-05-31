#[macro_use] extern crate hal;
extern crate arrayfire as af;

use hal::Model;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, SinSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Backend};


fn main() {
  // First we need to parameterize our network
  let input_dims = 128;
  let hidden_dims = 32;
  let output_dims = 128;
  let num_train_samples = 65536;
  let batch_size = 128;
  let optimizer_type = "SGD";
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
  model.add::<f32>("dense", hashmap!["activation"    => "tanh".to_string()
                                     , "input_size"  => hidden_dims.to_string()
                                     , "output_size" => output_dims.to_string()
                                     , "w_init"      => "glorot_uniform".to_string()
                                     , "b_init"      => "zeros".to_string()]);

  // Get some nice information about our model
  model.info();

  // Temporarily set the backend to CPU so that we can load data into RAM
  // The model will automatically toggle to the desired backend during training
  manager.swap_device(cpu_device);

  // Build our sin wave source
  let sin_generator = SinSource::new(input_dims, batch_size
                                     , num_train_samples
                                     , false   // normalized
                                     , false); // shuffled

  // Pull a sample to verify sizing
  let test_sample = sin_generator.get_train_iter(batch_size);
  println!("test sample shape: {:?}"
           , test_sample.input.into_inner().dims());

  // iterate our model in Verbose mode (printing loss)
  // Note: more manual control can be enacted by directly calling
  //       forward/backward & optimizer update
  let loss = model.fit::<SinSource, f32>(&sin_generator        // what data source to pull from
                                         , cpu_device          // source device
                                         , epochs, batch_size  // self explanatory :)
                                         , true);              // verbose

  // plot our loss on a 512x512 grid with the provided title
  plot_vec(loss, "Loss vs. Iterations", 512, 512);

  // infer on one test and plot the first sample (row) of the predictions
  let test_sample = sin_generator.get_test_iter(1).input.into_inner();
  let prediction = model.forward::<f32>(&test_sample
                                        , cpu_device // source device
                                        , cpu_device // destination device
                                        , false);    // not training
  println!("\nprediction shape: {:?} | backend = {:?}"
           , prediction.dims(), prediction.get_backend());
  plot_array(&af::flat(&prediction), "Model Inference", 512, 512);
}
