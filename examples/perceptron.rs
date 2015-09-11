extern crate hal;
extern crate nalgebra as na;

use na::DMat;
use hal::{Model, Layer};
use hal::optimizer::{Optimizer, SGD};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::layer::{Dense};
use hal::plot::Plot;

fn build_optimizer(name: &'static str) -> Result<Box<Optimizer>, HALError> {
  match name{
    "SGD" => Ok(Box::new(SGD::default())),
    _     => Err(HALError::UNKNOWN),
  }
}

fn generate_sin_wave(input_dims: usize, num_rows: usize) -> DMat<f32> {
  let mut waves = DMat::<f32>::new_zeros(num_rows, input_dims);
  let mut index: f32 = 0.0f32;
  let delta: f32 = 2.0*3.1415/(input_dims as f32);
  for r in 0..num_rows {
    for c in 0..input_dims {
      waves[(r, c)] = index.sin();
      index += delta;
    }
  }
  waves
}

fn main() {
  // First we need to parameterize our network
  let input_dims = 256;
  let hidden_dims = 128;
  let output_dims = 256;
  let num_train_samples = 1024;
  let iter = 20;
  let batch_size = 128;
  let optimizer_type = "SGD";

  // Now, let's build a model with an optimizer and a loss function
  let mut model = Box::new(Sequential::new(build_optimizer(optimizer_type).unwrap(), "mse"));

  // Let's add a few layers why don't we? 
  model.add(Box::new(Dense::new(input_dims, hidden_dims, "tanh", "normal", "ones")));
  model.add(Box::new(Dense::new(hidden_dims, output_dims, "tanh", "normal", "ones")));

  // Get some nice information about our model
  model.info();

  // Test with learning to predict sin wave
  let data = generate_sin_wave(input_dims, num_train_samples);
  
  // iterate our model in Verbose mode (printing loss)
  let (loss, prediction) = model.fit(&data, &data, batch_size, iter, true, false);

  // plot our loss
  Plot(loss, "Loss vs. Iterations", 512, 512);
}
