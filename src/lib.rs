extern crate arrayfire as af;
extern crate itertools;
extern crate rand;
extern crate csv;
extern crate num;
extern crate statistical;
extern crate rustc_serialize;

pub use layer::{Layer};
pub mod layer;

pub use model::{Model};
pub mod model;

pub use optimizer::{Optimizer};
pub mod optimizer;

pub mod params;
pub mod error;
pub mod loss;
pub mod activations;
pub mod initializations;
pub mod plot;
pub mod utils;
pub mod device;
