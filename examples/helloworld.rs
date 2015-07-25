extern crate hal;
extern crate docopt;
extern crate arrayfire as af;

use af::{Array, Dim4};
use docopt::Docopt;
use hal::Model;
use hal::Layer;
use hal::model::{Sequential};
use hal::layer::{Dense};

static USAGE: &'static str = "
Usage: 
  hal <input_csv> <output_csv>
";


fn main() {
  // let args = Docopt::new(USAGE)
  //                    .and_then(|dopt| dopt.parse())
  //                    .unwrap_or_else(|e| e.exit());
  // println!("{:?}", args);

  let input_dims = 512;
  let output_dims = 256;
  
  let mut model = Box::new(Sequential::new("L2", "SGD"));
  model.add(Box::new(Dense::new(input_dims, output_dims)));
  model.info();

  let dims = Dim4::new(&[input_dims, 1, 1, 1]);
  let a = af::randu(dims, af::Aftype::F32).unwrap();
  let inference = model.forward(&a);
  af::print(&inference);
}
