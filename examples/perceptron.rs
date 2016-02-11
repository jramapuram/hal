#[macro_use] extern crate hal;
extern crate arrayfire as af;

use hal::Model;
use hal::optimizer::{Optimizer, SGD};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Array, Dim4, Aftype, Backend};

fn build_optimizer(name: &str) -> Result<Box<Optimizer>, HALError> {
  match name{
    "SGD" => Ok(Box::new(SGD::default())),
    _     => Err(HALError::UNKNOWN),
  }
}

fn generate_sin_wave(input_dims: u64, num_rows: u64) -> Array {
  let dims = Dim4::new(&[input_dims * num_rows, 1, 1, 1]);
  let x = af::div(&af::sin(&af::range(dims, 0, Aftype::F32).unwrap()).unwrap()
                  , &input_dims, false).unwrap();
  let wave = af::sin(&x).unwrap();
  af::moddims(&wave, Dim4::new(&[num_rows, input_dims, 1, 1])).unwrap()
}

fn main() {
  // First we need to parameterize our network
  let input_dims = 64;
  let hidden_dims = 32;
  let output_dims = 64;
  let num_train_samples = 65536;
  let batch_size = 128;
  let optimizer_type = "SGD";

  // Now, let's build a model with an device manager on a specific device,
  // an optimizer and a loss function
  // DEFAULT is: OpenCL -> CUDA -> CPU
  let manager = DeviceManagerFactory::new();
  let gpu_device = Device{backend: Backend::AF_BACKEND_CUDA, id: 0};
  let cpu_device = Device{backend: Backend::AF_BACKEND_CPU, id: 0};
  let mut model = Box::new(Sequential::new(manager.clone()
                                           , build_optimizer(optimizer_type).unwrap()   // optimizer
                                           , "mse"                                      // loss
                                           , gpu_device));                              // device for model

  // Let's add a few layers why don't we?
  model.add("dense", hashmap!["activation"    => "tanh".to_string()
                              , "input_size"  => input_dims.to_string()
                              , "output_size" => hidden_dims.to_string()
                              , "w_init"      => "glorot_uniform".to_string()
                              , "b_init"      => "zeros".to_string()]);
  model.add("dense", hashmap!["activation"    => "tanh".to_string()
                              , "input_size"  => hidden_dims.to_string()
                              , "output_size" => output_dims.to_string()
                              , "w_init"      => "glorot_uniform".to_string()
                              , "b_init"      => "zeros".to_string()]);

  // Get some nice information about our model
  model.info();

  // Temporarily set the backend to CPU so that we can load data into RAM
  // The model will automatically toggle to the desired backend during training
  manager.swap_device(cpu_device);

  // Test with learning to predict sin wave
  let mut train = generate_sin_wave(input_dims, num_train_samples);
  let mut test = generate_sin_wave(input_dims, batch_size);
  let mut target = train.clone();

  // iterate our model in Verbose mode (printing loss)
  let loss = model.fit(&mut train, &mut target
                       , cpu_device, batch_size
                       , false  // shuffle
                       , true); // verbose

  // plot our loss
  plot_vec(loss, "Loss vs. Iterations", 512, 512);

  // infer on one of our samples
  println!("test shape= {:?}", test.dims().unwrap().get().clone());
  println!("train shape= {:?}", train.dims().unwrap().get().clone());
  let prediction = model.forward(&test, cpu_device, false);
  println!("prediction shape: {:?}", prediction.dims().unwrap().get().clone());
  plot_array(&af::flat(&af::rows(&prediction, 0, 1).unwrap()).unwrap(), "Model Inference", 512, 512);
}
