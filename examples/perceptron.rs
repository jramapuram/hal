extern crate hal;
extern crate docopt;
extern crate arrayfire as af;

use af::{Array, Dim4};
use hal::{Model, Layer};
use hal::optimizer::{Optimizer, SGD};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::layer::{Dense};


fn build_optimizer(name: &'static str) -> Result<Box<Optimizer>, HALError> {
  match name{
    "SGD" => Ok(Box::new(SGD::default())),
    _     => Err(HALError::UNKNOWN),
  }
}

fn generate_sin_wave(input_dims: u64, num_rows: u64) -> Vec<Array> {
  let dims = Dim4::new(&[input_dims, 1, 1, 1]);
  let delta = 2.0*3.1415/(input_dims as f32 - 1.0);

  let index = af::range(dims, 0, af::Aftype::F32).unwrap();
  let mut range = af::mul(&index, &delta).unwrap();
  let mut waves = Vec::<Array>::new();
  
  for i in (input_dims..num_rows+input_dims) {
    waves.push(af::sin(&range).unwrap());
    range = af::mul(&af::add(&index, &(1.0 as f32)).unwrap(), &delta).unwrap();
  }
  waves
}

fn main() {
  // First we need to parameterize our network
  let input_dims = 8;
  let hidden_dims = 4;
  let output_dims = 8;
  let num_train_samples = 1024;
  let iter = 200;
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
  
  // iterate our model
  let (loss, prediction) = model.fit(&data, &data, batch_size, iter, true);
  println!("prediction: "); af::print(&prediction);
  println!("loss: {:?}", loss);
}
