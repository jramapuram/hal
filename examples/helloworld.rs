extern crate hal;
extern crate docopt;

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

  let mut model = Box::new(Sequential::new("L2", "SGD"));
  model.add(Box::new(Dense::new(512, 256)));
  model.info();
}
