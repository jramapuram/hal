#[macro_use] extern crate hal;
extern crate arrayfire as af;

use hal::Model;
use hal::optimizer::{Optimizer, SGD};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use af::{Array, Dim4, AfBackend, Aftype};

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
  let batch_size = 32;
  let optimizer_type = "SGD";

  // Now, let's build a model with an optimizer and a loss function
  let mut model = Box::new(Sequential::new(build_optimizer(optimizer_type).unwrap() //optimizer
                                           , "mse"                                  // loss
                                           , AfBackend::AF_BACKEND_CUDA             // backend
                                           , 0));                                   // device_id

  // Let's add a few layers why don't we?
  let input_str: &str = &input_dims.to_string();
  let hidden_str: &str = &hidden_dims.to_string();
  let output_str: &str = &output_dims.to_string();
  model.add("dense", hashmap!["activation"    => "tanh"
                              , "input_size"  => input_str//&input_dims.to_string()
                              , "output_size" => hidden_str//&hidden_dims.to_string()
                              , "w_init"      => "glorot_uniform"
                              , "b_init"      => "zeros"]);
  model.add("dense", hashmap!["activation"    => "tanh"
                              , "input_size"  => hidden_str//&hidden_dims.to_string()
                              , "output_size" => output_str//&output_dims.to_string()
                              , "w_init"      => "glorot_uniform"
                              , "b_init"      => "zeros"]);

  // Get some nice information about our model
  model.info();

  // Temporarily set the backend to CPU so that we can load data into RAM
  // The model will automatically toggle to the desired backend during training
  model.set_device(AfBackend::AF_BACKEND_CPU, 0);

  // Test with learning to predict sin wave
  let mut data = generate_sin_wave(input_dims, num_train_samples);
  let mut target = data.clone();

  // iterate our model in Verbose mode (printing loss)
  let loss = model.fit(&mut data, &mut target, batch_size
                       , false  // shuffle
                       , true); // verbose


  // plot our loss
  plot_vec(loss, "Loss vs. Iterations", 512, 512);

  // infer on one of our samples
  let temp = af::rows(&data, 0, batch_size - 1).unwrap();
  println!("temp shape= {:?}", temp.dims().unwrap().get().clone());
  let prediction = model.forward(&af::rows(&data, 0, batch_size - 1).unwrap(), false);
  println!("prediction shape: {:?}", prediction.dims().unwrap().get().clone());
  plot_array(&prediction, "Model Inference", 512, 512);
}
