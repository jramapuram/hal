extern crate hal;
extern crate nalgebra as na;

use hal::utils;
use na::{DMat, ColSlice, Shape};
use hal::{Model, Layer};
use hal::optimizer::{Optimizer, SGD};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::layer::{Dense};
use hal::plot::{plot_vec, plot_dvec};

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
  let input_dims = 64;
  let hidden_dims = 32;
  let output_dims = 64;
  let num_train_samples = 65536;
  let batch_size = 16;
  let optimizer_type = "SGD";

  // Now, let's build a model with an optimizer and a loss function
  let mut model = Box::new(Sequential::new(build_optimizer(optimizer_type).unwrap(), "mse"));

  // Set which GPU we want to run this model on
  model.set_device(0);

  // Let's add a few layers why don't we?
  model.add(Box::new(Dense::new(input_dims, hidden_dims, "tanh", "glorot_uniform", "zeros")));
  model.add(Box::new(Dense::new(hidden_dims, output_dims, "tanh", "glorot_uniform", "zeros")));

  // Get some nice information about our model
  model.info();

  // Test with learning to predict sin wave
  let mut data = generate_sin_wave(input_dims as usize, num_train_samples);
  let mut target = data.clone();

  // iterate our model in Verbose mode (printing loss)
  let (loss, prediction) = model.fit(&mut data, &mut target, batch_size, true, true);

  // plot our loss
  plot_vec(loss, "Loss vs. Iterations", 512, 512);
  println!("pred shape: {:?}", prediction.shape());
  let pred_rows = prediction.nrows();
  plot_dvec(&prediction.col_slice(0, 0, pred_rows - 1), "Final Prediction Rows", 512, 512);

  // write one of the results to csv
  utils::write_csv::<f32>("prediction_col.csv", &prediction.col_slice(0, 0, pred_rows - 1).at);
}
