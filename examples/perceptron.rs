extern crate hal;
extern crate docopt;
extern crate arrayfire as af;

use docopt::Docopt;
use af::{Array, Dim4};
use hal::{Model, Layer};
use hal::optimizer::{Optimizer, SGD};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::layer::{Dense};

static USAGE: &'static str = "
Usage: 
  hal <input_csv> <output_csv>
";

fn build_optimizer(name: &'static str) -> Result<Box<Optimizer>, HALError>{
  match name{
    "SGD" => Ok(Box::new(SGD::default())),
    _     => Err(HALError::UNKNOWN),
  }
}

fn main() {
  // let args = Docopt::new(USAGE)
  //                    .and_then(|dopt| dopt.parse())
  //                    .unwrap_or_else(|e| e.exit());
  // println!("{:?}", args);

  let input_dims = 512;
  let hidden_dims = 256;
  let output_dims = 512;
  let optimizer_type = "SGD";

  let mut model = Box::new(Sequential::new(build_optimizer(optimizer_type).unwrap(), "mse"));
  model.add(Box::new(Dense::new(input_dims, hidden_dims, "tanh", "normal", "ones")));
  model.add(Box::new(Dense::new(hidden_dims, output_dims, "tanh", "normal", "ones")));
  model.info();

  let dims = Dim4::new(&[input_dims, 1, 1, 1]);
  let original = af::randu(dims, af::Aftype::F32).unwrap();
  let inference = model.forward(&original);
  let (loss, prediction) = model.backward(&inference, &original);
  af::print(&prediction);
  println!("loss: {}", loss);
}
